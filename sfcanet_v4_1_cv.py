"""
SFCANet-v4.1 Pilot Run (Fold 0) -- Stable Cross-Domain CV
=========================================================
Implements the 5-Fold Group CV pilot run with:
  - Adaptive Frequency-Guided Attention (AFGA)
  - Segment-Aware Pooling
  - Fourier Perturbation
  - SWA (Stochastic Weight Averaging) & Gradient Clipping
"""

import os, gc, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR

from phase4_config import (
    DEVICE, SEED,
    PHASE4_CHECKPOINT_DIR, PHASE4_RESULTS_DIR,
    PHYSICAL_BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS,
    LEARNING_RATE, LR_MIN, WEIGHT_DECAY, WARMUP_EPOCHS,
    POS_WEIGHT, DCT_SCALES,
    HARD_NEG_MINING_EPOCH, HARD_NEG_TOP_FRACTION, HARD_NEG_WEIGHT_MULT,
)
from mixed_dataset import get_mixed_loaders_v4_1, mixed_collate_v4_1
from sfcanet import SFCANetV4
from sfcanet_v4_train import set_seed, build_scheduler, rebuild_sampler_with_hnm

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch: int) -> dict:
    model.train()
    clf_loss_sum = 0.0
    num_batches  = 0
    loss_by_idx  = {}
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        rgb    = batch['rgb'].to(DEVICE, non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct    = {s: batch['dct'][s].to(DEVICE, non_blocking=True) for s in DCT_SCALES}
        idxs   = batch.get('idx', None)

        with autocast('cuda'):
            logit = model(rgb, dct)
            clf_loss = criterion(logit, labels) / ACCUM_STEPS

        scaler.scale(clf_loss).backward()

        if idxs is not None:
            with torch.no_grad():
                per_sample = nn.functional.binary_cross_entropy_with_logits(
                    logit, labels, reduction='none'
                ).squeeze(1).cpu().numpy()
            for i, idx in enumerate(idxs):
                loss_by_idx[idx] = float(per_sample[i])

        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        clf_loss_sum += clf_loss.item() * ACCUM_STEPS
        num_batches  += 1

    if (batch_idx + 1) % ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return {
        'clf_loss':   clf_loss_sum / max(num_batches, 1),
        'loss_by_idx': loss_by_idx,
    }

@torch.no_grad()
def evaluate_segment_aware(model, loader, criterion, name: str) -> dict:
    """Segment-Aware Pooling evaluation."""
    model.eval()
    segment_probs = {}
    segment_labels = {}
    loss_sum = 0.0

    for batch in loader:
        rgb    = batch['rgb'].to(DEVICE, non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True).unsqueeze(1)
        dct    = {s: batch['dct'][s].to(DEVICE, non_blocking=True) for s in DCT_SCALES}
        seg_ids = batch['segment_id']

        with autocast('cuda'):
            logit = model(rgb, dct)
            loss  = criterion(logit, labels)

        loss_sum += loss.item()
        probs = torch.sigmoid(logit).cpu().squeeze()
        if probs.ndim == 0:
            probs = probs.unsqueeze(0)
            
        for i, seg_id in enumerate(seg_ids):
            if seg_id not in segment_probs:
                segment_probs[seg_id] = []
                segment_labels[seg_id] = labels[i].cpu().item()
            segment_probs[seg_id].append(probs[i].item())

    # Average over segments (Median Pooling to filter burst noise)
    final_probs, final_labels = [], []
    for seg_id, p_list in segment_probs.items():
        final_probs.append(float(np.median(p_list)))
        final_labels.append(segment_labels[seg_id])

    avg_loss = loss_sum / max(len(loader), 1)
    try:
        auc = roc_auc_score(final_labels, final_probs)
    except ValueError:
        auc = 0.0

    preds = (np.array(final_probs) >= 0.5).astype(int)
    acc   = (preds == np.array(final_labels)).mean()
    print(f"    [{name:14s}] Loss: {avg_loss:.4f} | Seg-Acc: {acc:.4f} | Seg-AUC: {auc:.4f}")
    
    return {'loss': avg_loss, 'acc': acc, 'auc': auc}

def main(fold_idx=0):
    print("=" * 70)
    print(f"  SFCANet-v4.1 Run (Fold {fold_idx}) - Stable Cross-Domain")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {NUM_EPOCHS} | Eff. batch: {PHYSICAL_BATCH_SIZE * ACCUM_STEPS}")
    print(f"  LR     : {LEARNING_RATE} | HNM at {HARD_NEG_MINING_EPOCH} (x{HARD_NEG_WEIGHT_MULT})")
    print("=" * 70)

    set_seed(SEED + fold_idx)

    # Load specific Fold
    train_ld, val_aigu_ld, val_ffpp_ld, w_out, train_ds = get_mixed_loaders_v4_1(fold_idx=fold_idx)

    model = SFCANetV4(use_gradient_checkpoint=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE))
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)
    scaler    = GradScaler('cuda')

    # SWA setup
    swa_model = AveragedModel(model)
    swa_start = 25  # Start SWA at Epoch 26
    swa_scheduler = SWALR(optimizer, swa_lr=LR_MIN)

    ckpt_path = os.path.join(PHASE4_CHECKPOINT_DIR, f"sfcanet_v4_1_cv{fold_idx}_best.pth")
    log_path  = os.path.join(PHASE4_RESULTS_DIR, f"sfcanet_v4_1_cv{fold_idx}_log.csv")
    
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'clf_loss', 'aigu_auc', 'ffpp_auc', 'combined_auc', 'lr', 'time_sec', 'hnm_active'])

    best_combined = 0.0
    loss_by_idx = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        hnm_active = (epoch >= HARD_NEG_MINING_EPOCH)

        # HNM needs base weights
        base_weights = w_out
        
        if hnm_active and loss_by_idx:
            new_sampler = rebuild_sampler_with_hnm(base_weights, loss_by_idx, n_samples=len(train_ds))
            train_ld = torch.utils.data.DataLoader(
                train_ds, batch_size=PHYSICAL_BATCH_SIZE, sampler=new_sampler, 
                num_workers=2, collate_fn=mixed_collate_v4_1, pin_memory=True, drop_last=True
            )

        train_stats = train_one_epoch(model, train_ld, criterion, optimizer, scaler, epoch)
        loss_by_idx = train_stats['loss_by_idx']

        print(f"\n  Epoch {epoch:2d}/{NUM_EPOCHS}{'  [HNM ON]' if hnm_active else ''}{'  [SWA]' if epoch > swa_start else ''}")
        
        val_aigu = evaluate_segment_aware(model, val_aigu_ld, criterion, "AIGuard-val")
        val_ffpp = evaluate_segment_aware(model, val_ffpp_ld, criterion, "FF++-val")

        combined_auc = (val_aigu['auc'] * val_ffpp['auc']) ** 0.5
        
        # SWA Schedule update
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        if combined_auc > best_combined:
            best_combined = combined_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_combined_auc': best_combined}, ckpt_path)

        print(f"    Combined AUC: {combined_auc:.4f} | CLF: {train_stats['clf_loss']:.4f} | LR: {lr:.2e} | {dt:.0f}s")
        
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_stats['clf_loss']:.6f}", f"{val_aigu['auc']:.6f}", f"{val_ffpp['auc']:.6f}", f"{combined_auc:.6f}", f"{lr:.8f}", f"{dt:.1f}", int(hnm_active)])

    print("\n  [SWA] Updating BatchNorm stats for averaged model...")
    try:
        swa_model.train()
        for batch in train_ld:
            rgb = batch['rgb'].to(DEVICE)
            dct = {s: batch['dct'][s].to(DEVICE) for s in DCT_SCALES}
            swa_model(rgb, dct)
    except Exception as e:
        print(f"  [SWA] BN update skipped: {e}")

    print("\n  [SWA] Final Evaluation:")
    val_ffpp_swa = evaluate_segment_aware(swa_model, val_ffpp_ld, criterion, "FF++-SWA-val")
    print(f"  Final SWA FF++ AUC: {val_ffpp_swa['auc']:.4f}")

    print(f"  [DONE] Fold {fold_idx} Complete. Best Epoch AUC: {best_combined:.4f}")

if __name__ == "__main__":
    main(fold_idx=0)
