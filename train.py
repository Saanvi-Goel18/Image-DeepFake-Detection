"""
Step 4: Training Pipeline
==========================
- Mixed Precision (AMP) for VRAM efficiency
- Gradient Accumulation (×4) → effective batch = 64
- AUC-ROC based checkpointing (saves best model per metric)
- CosineAnnealingLR scheduler
- Trains all 3 models sequentially

Usage:
    python 04_train.py
    python 04_train.py --model resnet50          # Train single model
    python 04_train.py --model efficientnet_b0   # Train single model
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

from config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR, MODEL_NAMES,
    PHYSICAL_BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, SEED
)
from dataset import get_dataloaders
from models import get_model


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps):
    """Train for one epoch with AMP + gradient accumulation."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accum_steps  # Normalize for accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Step optimizer every accum_steps
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps  # Undo normalization for logging
        num_batches += 1

    # Handle remaining gradients (if batches not divisible by accum_steps)
    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss, accuracy, AUC-ROC, and predictions."""
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast():
            outputs = model(images)
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
        auc = 0.0  # If only one class present

    return avg_loss, accuracy, auc, all_probs, all_labels


def train_model(model_name, train_loader, val_loader):
    """Full training loop for a single model."""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name.upper()}")
    print(f"{'='*70}")

    set_seed(SEED)

    # Create model
    model = get_model(model_name).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler()

    # Checkpoint path
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")

    # Training log
    log_path = os.path.join(RESULTS_DIR, f"{model_name}_training_log.csv")
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_auc', 'lr', 'time_sec'
    ])

    best_auc = 0.0
    total_train_time = 0.0

    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
    print(f"  Batches/epoch: {len(train_loader)} (physical BS={PHYSICAL_BATCH_SIZE})")
    print(f"  Gradient accumulation: {ACCUM_STEPS} steps")
    print(f"  Effective batch size: {PHYSICAL_BATCH_SIZE * ACCUM_STEPS}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, ACCUM_STEPS
        )

        # Evaluate
        val_loss, val_acc, val_auc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        # Log
        log_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_acc:.4f}", f"{val_auc:.6f}", f"{current_lr:.8f}",
            f"{epoch_time:.1f}"
        ])
        log_file.flush()

        # Checkpoint (best AUC-ROC)
        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_name': model_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_accuracy': val_acc,
            }, best_ckpt_path)
            improved = " ★ BEST"

        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} │ "
              f"Train Loss: {train_loss:.4f} │ "
              f"Val Loss: {val_loss:.4f} │ "
              f"Val Acc: {val_acc:.4f} │ "
              f"Val AUC: {val_auc:.4f} │ "
              f"LR: {current_lr:.6f} │ "
              f"{epoch_time:.0f}s{improved}")

    log_file.close()

    print(f"\n  ✓ Training complete for {model_name}")
    print(f"    Best AUC-ROC: {best_auc:.4f}")
    print(f"    Total time: {total_train_time:.0f}s ({total_train_time/60:.1f}min)")
    print(f"    Checkpoint: {best_ckpt_path}")

    # Cleanup
    del model, optimizer, scheduler, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()

    return best_auc, total_train_time


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection baselines")
    parser.add_argument('--model', type=str, default=None,
                        help="Train specific model (e.g., resnet50). "
                             "Default: train all 3 models.")
    args = parser.parse_args()

    print("=" * 70)
    print("  DEEPFAKE DETECTION — TRAINING PIPELINE")
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
          if DEVICE.type == 'cuda' else "")
    print("=" * 70)

    # Load data
    print("\n[Data] Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Determine which models to train
    models_to_train = [args.model] if args.model else MODEL_NAMES

    results_summary = {}
    for model_name in models_to_train:
        best_auc, train_time = train_model(model_name, train_loader, val_loader)
        results_summary[model_name] = {
            'best_val_auc': best_auc,
            'train_time_sec': train_time
        }

    # Print summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    for name, res in results_summary.items():
        print(f"  {name:25s} │ Best AUC: {res['best_val_auc']:.4f} │ "
              f"Time: {res['train_time_sec']/60:.1f}min")
    print("=" * 70)
    print("\n  Run 05_evaluate.py to generate the comparison table.")


if __name__ == "__main__":
    main()
