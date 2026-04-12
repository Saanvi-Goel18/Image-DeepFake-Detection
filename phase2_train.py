"""
Phase 2 Training Pipeline
===========================
Trains: (1) Standalone DCT-CNN, (2) SFCANet-v2 (bidirectional, layer4 unfrozen).
AMP + Gradient Checkpointing for RTX 3050 (4GB VRAM).

Usage:
    python phase2_train.py                      # Train all
    python phase2_train.py --mode dct_only      # DCT-CNN alone (balanced)
    python phase2_train.py --mode sfcanet_v2    # SFCANet-v2 (imbalanced, both pos_weights)
"""

import os
import gc
import csv
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from phase2_config import (
    DEVICE, PHASE2_CHECKPOINT_DIR, PHASE2_RESULTS_DIR, PHASE1_RESNET50_CKPT,
    PHYSICAL_BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    POS_WEIGHTS, SEED
)
from phase2_dataset import get_phase2_loaders
from dct_cnn import DCTCNN
from sfcanet import SFCANet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    accum_steps, mode='sfcanet'):
    """Train one epoch with AMP + gradient accumulation."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        if mode == 'dct_only':
            _, dct_coeffs, labels = batch
            dct_coeffs = dct_coeffs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast():
                outputs = model(dct_coeffs)
                loss = criterion(outputs, labels) / accum_steps
        else:
            images, dct_coeffs, labels = batch
            images = images.to(device, non_blocking=True)
            dct_coeffs = dct_coeffs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast():
                outputs = model(images, dct_coeffs)
                loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        num_batches += 1

    # Handle remaining gradients
    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode='sfcanet'):
    """Evaluate and return loss, accuracy, AUC-ROC."""
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    correct = 0
    total = 0

    for batch in loader:
        if mode == 'dct_only':
            _, dct_coeffs, labels = batch
            dct_coeffs = dct_coeffs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast():
                outputs = model(dct_coeffs)
                loss = criterion(outputs, labels)
        else:
            images, dct_coeffs, labels = batch
            images = images.to(device, non_blocking=True)
            dct_coeffs = dct_coeffs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast():
                outputs = model(images, dct_coeffs)
                loss = criterion(outputs, labels)

        running_loss += loss.item()
        probs = torch.sigmoid(outputs).cpu()
        preds = (probs >= 0.5).float()
        all_probs.extend(probs.squeeze().tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())
        correct += (preds.squeeze() == labels.cpu().squeeze()).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    return avg_loss, accuracy, auc


def train_model(model, train_ld, val_ld, model_name, pos_weight, mode='sfcanet'):
    """Full training loop for a single model configuration."""
    pw_str = f"pw{pos_weight:.0f}"
    full_name = f"{model_name}_{pw_str}"

    print(f"\n{'='*70}")
    print(f"  TRAINING: {full_name.upper()}")
    print(f"  pos_weight = {pos_weight}")
    print(f"{'='*70}")

    set_seed(SEED)
    model = model.to(DEVICE)

    # Weighted BCE Loss
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    ckpt_path = os.path.join(PHASE2_CHECKPOINT_DIR, f"{full_name}_best.pth")
    log_path = os.path.join(PHASE2_RESULTS_DIR, f"{full_name}_training_log.csv")
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_auc', 'lr', 'time_sec'])

    best_auc = 0.0
    total_time = 0.0

    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Criterion: BCEWithLogitsLoss(pos_weight={pos_weight})")
    print(f"  Batches/epoch: {len(train_ld)} | Effective BS: {PHYSICAL_BATCH_SIZE * ACCUM_STEPS}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_ld, criterion, optimizer, scaler, DEVICE, ACCUM_STEPS, mode
        )
        val_loss, val_acc, val_auc = evaluate(model, val_ld, criterion, DEVICE, mode)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        total_time += dt

        log_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_acc:.4f}", f"{val_auc:.6f}", f"{lr:.8f}", f"{dt:.1f}"
        ])
        log_file.flush()

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_name': full_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'pos_weight': pos_weight,
            }, ckpt_path)
            improved = " ★ BEST"

        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} │ "
              f"Train: {train_loss:.4f} │ Val: {val_loss:.4f} │ "
              f"Acc: {val_acc:.4f} │ AUC: {val_auc:.4f} │ "
              f"LR: {lr:.6f} │ {dt:.0f}s{improved}")

    log_file.close()
    print(f"\n  ✓ {full_name} complete | Best AUC: {best_auc:.4f} | Time: {total_time/60:.1f}min")

    # Cleanup
    del optimizer, scheduler, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return best_auc, total_time


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Training")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'dct_only', 'sfcanet_v2'],
                        help="What to train: 'dct_only', 'sfcanet_v2', or 'all'")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 2 — COST-SENSITIVE FREQUENCY LEARNING (SFCANet-v2)")
    print(f"  Device: {DEVICE} "
          f"({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    print("=" * 70)

    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Standalone DCT-CNN on balanced data
    # ═══════════════════════════════════════════════════════════════════════
    if args.mode in ('all', 'dct_only'):
        print("\n" + "━" * 70)
        print("  STEP 1: Standalone DCT-CNN (Balanced Dataset)")
        print("━" * 70)

        train_ld, val_ld, _ = get_phase2_loaders(use_balanced=True)
        model = DCTCNN(standalone=True)
        auc, t = train_model(model, train_ld, val_ld, "dct_cnn_standalone",
                             pos_weight=1.0, mode='dct_only')
        results['dct_cnn_standalone_pw1'] = {'auc': auc, 'time': t}

        del model, train_ld, val_ld
        gc.collect()
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: SFCANet-v2 (bidirectional + unfrozen layer4) on imbalanced data
    # ═══════════════════════════════════════════════════════════════════════
    if args.mode in ('all', 'sfcanet_v2'):
        print("\n" + "━" * 70)
        print("  STEP 2: SFCANet-v2 (Imbalanced 1:9 Dataset)")
        print("  Changes: Bidirectional cross-attention + ResNet layer4 unfrozen")
        print("━" * 70)

        train_ld, val_ld, _ = get_phase2_loaders(use_balanced=False)

        for pw in POS_WEIGHTS:
            print(f"\n  >>> pos_weight = {pw}")
            ckpt = PHASE1_RESNET50_CKPT if os.path.exists(PHASE1_RESNET50_CKPT) else None
            model = SFCANet(resnet50_ckpt_path=ckpt, use_gradient_checkpoint=True,
                          unfreeze_layer4=True)
            auc, t = train_model(model, train_ld, val_ld, "sfcanet_v2",
                                 pos_weight=pw, mode='sfcanet')
            results[f'sfcanet_v2_pw{pw:.0f}'] = {'auc': auc, 'time': t}

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 2 TRAINING SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:35s} │ AUC: {r['auc']:.4f} │ Time: {r['time']/60:.1f}min")
    print("=" * 70)
    print("\n  Run phase2_evaluate.py to generate comparison tables.")


if __name__ == "__main__":
    main()
