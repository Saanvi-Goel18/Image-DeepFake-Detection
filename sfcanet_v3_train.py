"""
SFCANet-v3 Training — Cross-Domain Adversarial Training
=========================================================
Trains SFCANetV3 on a mixed AIGuard + FaceForensics++ dataset using:
  - Cost-sensitive BCE (pos_weight=5 to penalise missed fakes)
  - Gradient Reversal adversarial domain alignment (annealed λ)
  - Dual validation: AIGuard (in-domain) + FF++ (cross-domain)
  - Best checkpoint saved on COMBINED (AIGuard AUC + FF++ AUC) metric

Usage:
    py -3.11 sfcanet_v3_train.py
"""

import os
import gc
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from phase3_config import (
    DEVICE, SEED,
    PHASE3_CHECKPOINT_DIR, PHASE3_RESULTS_DIR,
    PHYSICAL_BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS,
    LEARNING_RATE, LR_MIN, WEIGHT_DECAY,
    POS_WEIGHT,
    GRL_LAMBDA_MAX, GRL_WARMUP_EPOCHS, DOMAIN_LOSS_WEIGHT,
    DCT_SCALES,
)
from mixed_dataset import get_mixed_loaders
from sfcanet import SFCANetV3


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── GRL Lambda Schedule ─────────────────────────────────────────────────────

def compute_grl_lambda(epoch: int, total_epochs: int,
                        warmup: int = GRL_WARMUP_EPOCHS,
                        max_lambda: float = GRL_LAMBDA_MAX) -> float:
    """
    Anneals λ from 0 to max_lambda after warmup epochs.
    Uses a smooth exponential schedule from DANN paper.
    """
    if epoch <= warmup:
        return 0.0
    progress = (epoch - warmup) / (total_epochs - warmup)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0))


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model: SFCANetV3,
                    loader,
                    clf_criterion,
                    dom_criterion,
                    optimizer,
                    scaler: GradScaler,
                    grl_lambda: float,
                    epoch: int) -> dict:
    """One training epoch with gradient accumulation."""
    model.train()
    model.set_grl_alpha(grl_lambda)

    clf_loss_sum  = 0.0
    dom_loss_sum  = 0.0
    num_batches   = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        rgb    = batch['rgb'].to(DEVICE,   non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        domain = batch['domain'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct    = {s: batch['dct'][s].to(DEVICE, non_blocking=True) for s in DCT_SCALES}

        with autocast('cuda'):
            main_logit, domain_logit = model(rgb, dct)

            clf_loss = clf_criterion(main_logit, labels) / ACCUM_STEPS
            dom_loss = dom_criterion(domain_logit, domain) / ACCUM_STEPS

            # Domain loss only kicks in after warmup
            total_loss = clf_loss + grl_lambda * DOMAIN_LOSS_WEIGHT * dom_loss

        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        clf_loss_sum += clf_loss.item() * ACCUM_STEPS
        dom_loss_sum += dom_loss.item() * ACCUM_STEPS
        num_batches  += 1

    # Flush remaining gradients
    if (batch_idx + 1) % ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return {
        'clf_loss': clf_loss_sum / max(num_batches, 1),
        'dom_loss': dom_loss_sum / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(model: SFCANetV3, loader, criterion, name: str) -> dict:
    """Evaluate classification performance on one loader."""
    model.eval()
    all_probs  = []
    all_labels = []
    loss_sum   = 0.0

    for batch in loader:
        rgb    = batch['rgb'].to(DEVICE,   non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct    = {s: batch['dct'][s].to(DEVICE, non_blocking=True) for s in DCT_SCALES}

        with autocast('cuda'):
            main_logit, _ = model(rgb, dct)
            loss = criterion(main_logit, labels)

        loss_sum += loss.item()
        probs = torch.sigmoid(main_logit).cpu().squeeze()
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
    print(f"    [{name:12s}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
    return {'loss': avg_loss, 'acc': acc, 'auc': auc}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SFCANet-v3 - Cross-Domain Adversarial Training")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Epochs : {NUM_EPOCHS} | Eff. batch: {PHYSICAL_BATCH_SIZE * ACCUM_STEPS}")
    print(f"  GRL    : warmup={GRL_WARMUP_EPOCHS} epochs -> lambda -> {GRL_LAMBDA_MAX}")
    print("=" * 70)

    set_seed(SEED)

    # ── Data ────────────────────────────────────────────────────────────────
    train_ld, val_aigu_ld, val_ffpp_ld, test_aigu_ld, test_ffpp_ld = get_mixed_loaders()
    print(f"\n  Train batches : {len(train_ld)}")
    print(f"  Val AGu batches: {len(val_aigu_ld)}")
    print(f"  Val FF++ batches: {len(val_ffpp_ld)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = SFCANetV3(grl_alpha=0.0, use_gradient_checkpoint=True).to(DEVICE)

    # ── Loss ─────────────────────────────────────────────────────────────────
    clf_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
    )
    dom_criterion = nn.BCEWithLogitsLoss()

    # ── Optimiser ────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN
    )
    scaler = GradScaler('cuda')

    # ── Logging ──────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(PHASE3_CHECKPOINT_DIR, "sfcanet_v3_best.pth")
    log_path  = os.path.join(PHASE3_RESULTS_DIR,    "sfcanet_v3_training_log.csv")

    log_file = open(log_path, 'w', newline='')
    log_w    = csv.writer(log_file)
    log_w.writerow(['epoch', 'clf_loss', 'dom_loss', 'grl_lambda',
                    'aigu_auc', 'ffpp_auc', 'combined_auc', 'lr', 'time_sec'])

    best_combined = 0.0
    total_time    = 0.0

    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Compute and set GRL lambda for this epoch
        grl_lambda = compute_grl_lambda(epoch, NUM_EPOCHS)
        model.set_grl_alpha(grl_lambda)

        # Train
        train_stats = train_one_epoch(
            model, train_ld, clf_criterion, dom_criterion,
            optimizer, scaler, grl_lambda, epoch
        )

        # Validate on both domains
        print(f"\n  Epoch {epoch:2d}/{NUM_EPOCHS}  (lambda_grl={grl_lambda:.3f})")
        val_aigu = evaluate(model, val_aigu_ld, clf_criterion, "AIGuard-val")
        val_ffpp = evaluate(model, val_ffpp_ld, clf_criterion, "FF++-val")

        # Combined metric: geometric mean of both AUCs
        combined_auc = (val_aigu['auc'] * val_ffpp['auc']) ** 0.5

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        total_time += dt

        # Save best checkpoint
        improved_flag = ""
        if combined_auc > best_combined:
            best_combined = combined_auc
            torch.save({
                'model_name':    'sfcanet_v3',
                'epoch':         epoch,
                'model_state_dict': model.state_dict(),
                'best_combined_auc': best_combined,
                'aigu_auc':      val_aigu['auc'],
                'ffpp_auc':      val_ffpp['auc'],
                'grl_lambda':    grl_lambda,
            }, ckpt_path)
            improved_flag = " [BEST]"

        print(f"    Combined AUC: {combined_auc:.4f}"
              f" (AIGuard={val_aigu['auc']:.4f}, FF++={val_ffpp['auc']:.4f})"
              f" | CLF: {train_stats['clf_loss']:.4f}"
              f" | DOM: {train_stats['dom_loss']:.4f}"
              f" | LR: {lr:.2e} | {dt:.0f}s{improved_flag}")

        log_w.writerow([
            epoch,
            f"{train_stats['clf_loss']:.6f}",
            f"{train_stats['dom_loss']:.6f}",
            f"{grl_lambda:.4f}",
            f"{val_aigu['auc']:.6f}",
            f"{val_ffpp['auc']:.6f}",
            f"{combined_auc:.6f}",
            f"{lr:.8f}",
            f"{dt:.1f}",
        ])
        log_file.flush()

    log_file.close()

    print("\n" + "=" * 70)
    print(f"  [DONE] SFCANet-v3 Training Complete")
    print(f"  Best Combined AUC : {best_combined:.4f}")
    print(f"  Total time        : {total_time / 60:.1f} min")
    print(f"  Checkpoint        : {ckpt_path}")
    print("=" * 70)

    # ── Final test evaluation ────────────────────────────────────────────────
    print("\n  Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("  Test Results:")
    test_aigu = evaluate(model, test_aigu_ld, clf_criterion, "AIGuard-test")
    test_ffpp = evaluate(model, test_ffpp_ld, clf_criterion, "FF++-test")
    print(f"\n  AIGuard Test AUC : {test_aigu['auc']:.4f}")
    print(f"  FF++    Test AUC : {test_ffpp['auc']:.4f}")
    print(f"  Combined Test AUC: {(test_aigu['auc'] * test_ffpp['auc'])**0.5:.4f}")

    # Save final metrics
    result_path = os.path.join(PHASE3_RESULTS_DIR, "sfcanet_v3_test_results.csv")
    with open(result_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'accuracy', 'auc'])
        w.writerow(['AIGuard', f"{test_aigu['acc']:.4f}", f"{test_aigu['auc']:.4f}"])
        w.writerow(['FF++',    f"{test_ffpp['acc']:.4f}", f"{test_ffpp['auc']:.4f}"])
    print(f"\n  Results saved to {result_path}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
