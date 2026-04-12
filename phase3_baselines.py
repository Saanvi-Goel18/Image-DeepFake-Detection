"""
Phase 3 Baselines
===================
Train baseline models (Xception, ConvNeXt-Tiny, Swin-T) on the imbalanced 
dataset and test their response to cost-sensitive learning (pos_weight).
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
import timm

from phase2_config import (
    DEVICE, PHASE2_CHECKPOINT_DIR, PHASE2_RESULTS_DIR,
    PHYSICAL_BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    POS_WEIGHTS, SEED
)
from phase2_dataset import get_phase2_loaders

class BaselineModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        print(f"[Baseline] Loading {model_name} from timm...")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1)

    def forward(self, images, dct_coeffs=None):
        # dct_coeffs are ignored by standard pure-RGB baselines
        return self.model(images)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps):
    model.train()
    running_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        images, dct_coeffs, labels = batch
        images = images.to(device, non_blocking=True)
        # We don't strictly need dct_coeffs for baselines but we pass them to match interface
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

    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / num_batches

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    correct = 0
    total = 0

    for batch in loader:
        images, dct_coeffs, labels = batch
        images = images.to(device, non_blocking=True)
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

def train_baseline(model_name, train_ld, val_ld, pos_weight):
    pw_str = f"pw{pos_weight:.0f}"
    
    # Sanitize model name for filename (e.g. Swin-T has tricky names)
    safe_name = model_name.replace('/', '_').replace('-', '_')
    full_name = f"{safe_name}_{pw_str}"

    print(f"\n{'='*70}")
    print(f"  TRAINING BASELINE: {full_name.upper()}")
    print(f"  pos_weight = {pos_weight}")
    print(f"{'='*70}")

    set_seed(SEED)
    model = BaselineModel(model_name).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(DEVICE)
    )

    # Use smaller LR for baselines if needed, but 1e-4 is standard
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    ckpt_path = os.path.join(PHASE2_CHECKPOINT_DIR, f"{full_name}_best.pth")
    
    best_auc = 0.0
    total_time = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_ld, criterion, optimizer, scaler, DEVICE, ACCUM_STEPS)
        val_loss, val_acc, val_auc = evaluate(model, val_ld, criterion, DEVICE)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        total_time += dt

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

    print(f"\n  ✓ {full_name} complete | Best AUC: {best_auc:.4f} | Time: {total_time/60:.1f}min")

    del model, optimizer, scheduler, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return best_auc, total_time

def main():
    print("=" * 70)
    print("  PHASE 3 — BASELINES Cost-Sensitive Learning")
    print("=" * 70)

    # 1:9 Imbalanced setup to properly match phase2 setup
    train_ld, val_ld, _ = get_phase2_loaders(use_balanced=False)

    baseline_architectures = [
        "xception",          
        "convnext_tiny",    
        "swin_tiny_patch4_window7_224"
    ]

    results = {}

    for model_name in baseline_architectures:
        for pw in POS_WEIGHTS:
            auc, t = train_baseline(model_name, train_ld, val_ld, pos_weight=pw)
            results[f"{model_name}_pw{pw}"] = {'auc': auc, 'time': t}

    print("\n" + "=" * 70)
    print("  BASELINES TRAINING SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:35s} │ AUC: {r['auc']:.4f} │ Time: {r['time']/60:.1f}min")
    print("=" * 70)

if __name__ == "__main__":
    main()
