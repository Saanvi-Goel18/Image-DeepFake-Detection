"""
SFCANet-v4 Training -- SOTA Cross-Domain Adversarial-Free Training
==================================================================
Trains SFCANetV4 on a 50/50 mixed AIGuard + FaceForensics++ dataset using:
  - Cost-sensitive BCE (pos_weight=5)
  - LR linear warmup (3 epochs) + CosineAnnealing
  - Hard Negative Mining (HNM) from epoch 12: top-20% high-loss samples
    get 5x sampling weight
  - Dual validation: AIGuard (in-domain) + FF++ (cross-domain)
    py -3.11 sfcanet_v4_train.py
"""

import os, gc, csv, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from phase4_config import (
    DEVICE, SEED,
    PHASE4_CHECKPOINT_DIR, PHASE4_RESULTS_DIR,
    PHYSICAL_BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS,
    LEARNING_RATE, LR_MIN, WEIGHT_DECAY, WARMUP_EPOCHS,
    POS_WEIGHT, DCT_SCALES,
    HARD_NEG_MINING_EPOCH, HARD_NEG_TOP_FRACTION, HARD_NEG_WEIGHT_MULT,
    TTA_N_AUGMENTS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)
from mixed_dataset import get_mixed_loaders_v4, mixed_collate
from sfcanet import SFCANetV4


# -- Reproducibility -----------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -- Schedulers ----------------------------------------------------------------


def build_scheduler(optimizer):
    """Linear warmup then CosineAnnealing using SequentialLR."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / WARMUP_EPOCHS,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - WARMUP_EPOCHS,
        eta_min=LR_MIN,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPOCHS],
    )


# -- HNM: Hard Negative Mining sampler rebuild ---------------------------------


def rebuild_sampler_with_hnm(base_weights: np.ndarray,
                              loss_by_idx: dict,
                              n_samples: int) -> WeightedRandomSampler:
    """
    Rebuilds the WeightedRandomSampler incorporating Hard Negative Mining.
    Samples that were in the top HARD_NEG_TOP_FRACTION by loss get
    HARD_NEG_WEIGHT_MULT x their base weight.
    """
    if not loss_by_idx:
        return WeightedRandomSampler(base_weights.tolist(),
                                     num_samples=n_samples, replacement=True)

    losses = np.zeros(len(base_weights), dtype=np.float32)
    for idx, loss in loss_by_idx.items():
        if idx < len(losses):
            losses[idx] = loss

    threshold = np.quantile(losses[losses > 0], 1.0 - HARD_NEG_TOP_FRACTION) \
        if (losses > 0).any() else float('inf')

    weights = base_weights.copy()
    hard_mask = losses >= threshold
    weights[hard_mask] *= HARD_NEG_WEIGHT_MULT

    print(f"    [HNM] Hard samples: {hard_mask.sum()} / {len(weights)} "
          f"(loss >= {threshold:.4f}), weight x{HARD_NEG_WEIGHT_MULT}")
    return WeightedRandomSampler(weights.tolist(),
                                 num_samples=n_samples, replacement=True)


# -- Training Loop -------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    epoch: int) -> dict:
    model.train()
    clf_loss_sum = 0.0
    num_batches  = 0
    loss_by_idx  = {}    # {dataset_idx: per-sample loss} for HNM
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        rgb    = batch['rgb'].to(DEVICE,   non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct    = {s: batch['dct'][s].to(DEVICE, non_blocking=True)
                  for s in DCT_SCALES}
        idxs   = batch.get('idx', None)    # for HNM tracking

        with autocast('cuda'):
            logit = model(rgb, dct)
            # full batch loss (for reduction)
            clf_loss = criterion(logit, labels) / ACCUM_STEPS

        scaler.scale(clf_loss).backward()

        # Track per-sample losses for HNM (unreduced)
        if idxs is not None:
            with torch.no_grad():
                per_sample = nn.functional.binary_cross_entropy_with_logits(
                    logit, labels, reduction='none'
                ).squeeze(1).cpu().numpy()
            for i, idx in enumerate(idxs.tolist()):
                loss_by_idx[idx] = float(per_sample[i])

        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        clf_loss_sum += clf_loss.item() * ACCUM_STEPS
        num_batches  += 1

    # Flush remaining gradient
    if (batch_idx + 1) % ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return {
        'clf_loss':   clf_loss_sum / max(num_batches, 1),
        'loss_by_idx': loss_by_idx,
    }


# -- Evaluation ----------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, criterion, name: str) -> dict:
    """Standard single-pass evaluation."""
    model.eval()
    all_probs, all_labels = [], []
    loss_sum = 0.0

    for batch in loader:
        rgb    = batch['rgb'].to(DEVICE,   non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct    = {s: batch['dct'][s].to(DEVICE, non_blocking=True)
                  for s in DCT_SCALES}

        with autocast('cuda'):
            logit = model(rgb, dct)
            loss  = criterion(logit, labels)

        loss_sum += loss.item()
        probs = torch.sigmoid(logit).cpu().squeeze()
        if probs.ndim == 0:
            probs = probs.unsqueeze(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())

    avg_loss = loss_sum / max(len(loader), 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc   = (preds == np.array(all_labels)).mean()
    print(f"    [{name:14s}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
    return {'loss': avg_loss, 'acc': acc, 'auc': auc,
            'probs': all_probs, 'labels': all_labels}


@torch.no_grad()
def evaluate_with_tta(model, loader, criterion, name: str,
                      n_augments: int = TTA_N_AUGMENTS) -> dict:
    """
    Test-Time Augmentation evaluation.
    Runs each batch through n_augments random augmented views and
    averages the probabilities -- typically +1-3% AUC vs single pass.
    """
    model.eval()
    tta_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_probs_avg, all_labels = [], []
    loss_sum = 0.0

    for batch in loader:
        rgb_orig = batch['rgb'].to(DEVICE, non_blocking=True)
        labels   = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct      = {s: batch['dct'][s].to(DEVICE, non_blocking=True)
                    for s in DCT_SCALES}

        # Aggregate over n_augments views
        prob_accum = torch.zeros(rgb_orig.size(0), 1, device=DEVICE)
        with autocast('cuda'):
            for _ in range(n_augments):
                # Apply TTA transform to each image in batch
                rgb_aug = torch.stack([
                    tta_tf(img) for img in rgb_orig.cpu()
                ]).to(DEVICE)
                prob_accum += torch.sigmoid(model(rgb_aug, dct))
            probs_avg = prob_accum / n_augments
            loss = criterion(torch.logit(probs_avg.clamp(1e-6, 1 - 1e-6)),
                             labels)

        loss_sum += loss.item()
        p = probs_avg.cpu().squeeze()
        if p.ndim == 0:
            p = p.unsqueeze(0)
        all_probs_avg.extend(p.tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())

    try:
        auc = roc_auc_score(all_labels, all_probs_avg)
    except ValueError:
        auc = 0.0
    preds = (np.array(all_probs_avg) >= 0.5).astype(int)
    acc   = (preds == np.array(all_labels)).mean()
    print(f"    [{name:14s}] TTA({n_augments}) Acc: {acc:.4f} | AUC: {auc:.4f}")
    return {'loss': loss_sum / max(len(loader), 1),
            'acc': acc, 'auc': auc,
            'probs': all_probs_avg, 'labels': all_labels}


# -- EER Threshold Calibration -------------------------------------------------


def find_eer_threshold(probs: list, labels: list) -> float:
    """
    Find the decision threshold at Equal Error Rate (EER) on a validation set.
    EER is where FPR == FNR. Used for calibrating binary decision boundary.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    # EER is where |FPR - FNR| is minimised
    eer_idx = np.argmin(np.abs(fpr - fnr))
    threshold = float(thresholds[eer_idx])
    eer       = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    print(f"    [EER Calibration] Threshold: {threshold:.4f} | EER: {eer:.4f}")
    return threshold


# -- Main ----------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  SFCANet-v4 - SOTA Cross-Domain Training")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Epochs : {NUM_EPOCHS} | Eff. batch: {PHYSICAL_BATCH_SIZE * ACCUM_STEPS}")
    print(f"  LR     : warmup {WARMUP_EPOCHS} ep -> {LEARNING_RATE} -> cosine -> {LR_MIN}")
    print(f"  HNM    : activates at epoch {HARD_NEG_MINING_EPOCH}")
    print("=" * 70)

    set_seed(SEED)

    # -- Data -----------------------------------------------------------------

    (train_ld, val_aigu_ld, val_ffpp_ld,
     test_aigu_ld, test_ffpp_ld,
     base_weights, train_ds) = get_mixed_loaders_v4()

    print(f"\n  Train batches : {len(train_ld)}")
    print(f"  Val AGu       : {len(val_aigu_ld)}")
    print(f"  Val FF++      : {len(val_ffpp_ld)}\n")

    # -- Model -----------------------------------------------------------------

    model = SFCANetV4(use_gradient_checkpoint=True).to(DEVICE)

    # -- Loss ------------------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
    )

    # -- Optimiser -------------------------------------------------------------

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)
    scaler    = GradScaler('cuda')

    # -- Logging ---------------------------------------------------------------

    ckpt_path = os.path.join(PHASE4_CHECKPOINT_DIR, "sfcanet_v4_best.pth")
    log_path  = os.path.join(PHASE4_RESULTS_DIR,    "sfcanet_v4_training_log.csv")

    log_file = open(log_path, 'w', newline='')
    log_w    = csv.writer(log_file)
    log_w.writerow(['epoch', 'clf_loss', 'aigu_auc', 'ffpp_auc',
                    'combined_auc', 'lr', 'time_sec', 'hnm_active'])

    best_combined = 0.0
    total_time    = 0.0
    loss_by_idx   = {}    # accumulated per-sample losses for HNM

    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        hnm_active = (epoch >= HARD_NEG_MINING_EPOCH)

        # Rebuild sampler with HNM from epoch HARD_NEG_MINING_EPOCH
        if hnm_active and loss_by_idx:
            new_sampler = rebuild_sampler_with_hnm(
                base_weights, loss_by_idx, n_samples=len(train_ds))
            # Rebuild loader with updated sampler
            num_workers = 2
            train_ld = torch.utils.data.DataLoader(
                train_ds, batch_size=PHYSICAL_BATCH_SIZE,
                sampler=new_sampler, num_workers=num_workers,
                collate_fn=mixed_collate, pin_memory=True,
                drop_last=True, persistent_workers=(num_workers > 0),
            )

        # Train
        train_stats = train_one_epoch(
            model, train_ld, criterion, optimizer, scaler, epoch)
        loss_by_idx = train_stats['loss_by_idx']

        # Validate
        print(f"\n  Epoch {epoch:2d}/{NUM_EPOCHS}"
              f"{'  [HNM ON]' if hnm_active else ''}")
        val_aigu = evaluate(model, val_aigu_ld, criterion, "AIGuard-val")
        val_ffpp = evaluate(model, val_ffpp_ld, criterion, "FF++-val")

        combined_auc = (val_aigu['auc'] * val_ffpp['auc']) ** 0.5
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        total_time += dt

        improved_flag = ""
        if combined_auc > best_combined:
            best_combined = combined_auc
            torch.save({
                'model_name':    'sfcanet_v4',
                'epoch':         epoch,
                'model_state_dict': model.state_dict(),
                'best_combined_auc': best_combined,
                'aigu_auc':      val_aigu['auc'],
                'ffpp_auc':      val_ffpp['auc'],
            }, ckpt_path)
            improved_flag = " [BEST]"

        print(f"    Combined AUC: {combined_auc:.4f}"
              f" (AIGuard={val_aigu['auc']:.4f}, FF++={val_ffpp['auc']:.4f})"
              f" | CLF: {train_stats['clf_loss']:.4f}"
              f" | LR: {lr:.2e} | {dt:.0f}s{improved_flag}")

        log_w.writerow([
            epoch,
            f"{train_stats['clf_loss']:.6f}",
            f"{val_aigu['auc']:.6f}",
            f"{val_ffpp['auc']:.6f}",
            f"{combined_auc:.6f}",
            f"{lr:.8f}",
            f"{dt:.1f}",
            int(hnm_active),
        ])
        log_file.flush()

    log_file.close()

    print("\n" + "=" * 70)
    print(f"  [DONE] SFCANet-v4 Training Complete")
    print(f"  Best Combined AUC : {best_combined:.4f}")
    print(f"  Total time        : {total_time / 60:.1f} min")
    print("=" * 70)

    # -- Final test evaluation with TTA ----------------------------------------

    print("\n  Loading best checkpoint for TTA test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # Phase 1: Find EER threshold on val set
    print("\n  EER Threshold Calibration on Val Set:")
    val_aigu_full = evaluate(model, val_aigu_ld, criterion, "AIGuard-val-cal")
    eer_thresh = find_eer_threshold(val_aigu_full['probs'],
                                    val_aigu_full['labels'])

    # Phase 2: TTA Test evaluation
    print("\n  TTA Test Results:")
    test_aigu = evaluate_with_tta(model, test_aigu_ld, criterion, "AIGuard-test")
    test_ffpp = evaluate_with_tta(model, test_ffpp_ld, criterion, "FF++-test")

    # Phase 3: Re-evaluate with EER threshold
    def eval_at_thresh(probs, labels, thresh, name):
        preds = (np.array(probs) >= thresh).astype(int)
        acc   = (preds == np.array(labels)).mean()
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0
        print(f"    [{name:14s}] EER-threshold acc: {acc:.4f} | AUC: {auc:.4f}")
        return acc, auc

    print(f"\n  Final Results (EER threshold = {eer_thresh:.4f}):")
    acc_ag, auc_ag = eval_at_thresh(test_aigu['probs'], test_aigu['labels'],
                                    eer_thresh, "AIGuard-test")
    acc_ff, auc_ff = eval_at_thresh(test_ffpp['probs'], test_ffpp['labels'],
                                    eer_thresh, "FF++-test")

    combined_test = (auc_ag * auc_ff) ** 0.5
    print(f"\n  AIGuard Test AUC  : {auc_ag:.4f}")
    print(f"  FF++    Test AUC  : {auc_ff:.4f}")
    print(f"  Combined Test AUC : {combined_test:.4f}")

    # Save final results
    result_path = os.path.join(PHASE4_RESULTS_DIR, "sfcanet_v4_test_results.csv")
    with open(result_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'accuracy', 'auc', 'eer_threshold'])
        w.writerow(['AIGuard', f"{acc_ag:.4f}", f"{auc_ag:.4f}",
                    f"{eer_thresh:.4f}"])
        w.writerow(['FF++',    f"{acc_ff:.4f}", f"{auc_ff:.4f}",
                    f"{eer_thresh:.4f}"])
    print(f"\n  Results saved -> {result_path}")


    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
