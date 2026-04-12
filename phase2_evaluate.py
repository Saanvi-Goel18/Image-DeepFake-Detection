"""
Phase 2 Evaluation — Punishment Parameter Analysis
=====================================================
Evaluates: DCT-CNN standalone, SFCANet (pw=1.0), SFCANet (pw=5.0)
Generates: Comparison table, confusion matrices, punishment effect analysis.

Usage:
    python phase2_evaluate.py
"""

import os
import csv
import time
import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn

from phase2_config import (
    DEVICE, PHASE2_CHECKPOINT_DIR, PHASE2_RESULTS_DIR, PHASE1_RESNET50_CKPT,
    PHYSICAL_BATCH_SIZE, POS_WEIGHTS
)
from phase2_dataset import get_phase2_loaders
from dct_cnn import DCTCNN
from sfcanet import SFCANet
import timm

class BaselineModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Pretrained = False because we are loading from a checkpoint later anyway
        self.model = timm.create_model(model_name, pretrained=False, num_classes=1)

    def forward(self, images, dct_coeffs=None):
        return self.model(images)


def load_model(model_type, ckpt_name):
    """Load a Phase 2 checkpoint."""
    path = os.path.join(PHASE2_CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(path):
        print(f"[Evaluate] Checkpoint not found: {path}")
        return None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    if model_type == 'dct_cnn':
        model = DCTCNN(standalone=True)
    elif model_type in ('sfcanet', 'sfcanet_v2'):
        rckpt = PHASE1_RESNET50_CKPT if os.path.exists(PHASE1_RESNET50_CKPT) else None
        model = SFCANet(resnet50_ckpt_path=rckpt, use_gradient_checkpoint=False,
                        unfreeze_layer4=(model_type == 'sfcanet_v2'))
    elif model_type in ('xception', 'convnext_tiny', 'swin_tiny_patch4_window7_224'):
        model = BaselineModel(model_type)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(DEVICE).eval()
    print(f"[Evaluate] Loaded {ckpt['model_name']} (epoch {ckpt['epoch']}, "
          f"AUC={ckpt['best_auc']:.4f}, pw={ckpt.get('pos_weight', 'N/A')})")
    return model


@torch.no_grad()
def run_inference(model, loader, mode='sfcanet'):
    """Run inference and return labels, probs, and inference time."""
    model.eval()
    all_probs = []
    all_labels = []
    total_time = 0.0
    total_samples = 0

    for batch in loader:
        if mode == 'dct_only':
            _, dct_coeffs, labels = batch
            dct_coeffs = dct_coeffs.to(DEVICE, non_blocking=True)
            bs = dct_coeffs.size(0)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast():
                outputs = model(dct_coeffs)
        else:
            images, dct_coeffs, labels = batch
            images = images.to(DEVICE, non_blocking=True)
            dct_coeffs = dct_coeffs.to(DEVICE, non_blocking=True)
            bs = images.size(0)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast():
                outputs = model(images, dct_coeffs)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        total_samples += bs

        probs = torch.sigmoid(outputs).cpu().squeeze().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_labels.extend(labels.numpy().tolist())

    inf_ms = (total_time / total_samples) * 1000
    return np.array(all_labels), np.array(all_probs), inf_ms


def compute_all_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'Accuracy': accuracy_score(labels, preds) * 100,
        'Precision': precision_score(labels, preds, zero_division=0) * 100,
        'Recall': recall_score(labels, preds, zero_division=0) * 100,
        'F1-Score': f1_score(labels, preds, zero_division=0) * 100,
        'AUC-ROC': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def plot_confusion_matrix(labels, probs, name, save_path, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(roc_data, save_path):
    plt.figure(figsize=(8, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800',
              '#00BCD4', '#E91E63', '#8BC34A', '#673AB7', '#795548', '#607D8B']
    for (name, fpr, tpr, auc_val), color in zip(roc_data, colors):
        plt.plot(fpr, tpr, color=color, linewidth=2.0,
                 label=f'{name} (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Phase 2 - ROC Curves (All Models)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] ROC curves saved: {save_path}")


def plot_punishment_comparison(results, save_path):
    """Bar chart comparing FN and Recall across pos_weight values."""
    sfca_results = [(n, m) for n, m, _ in results if 'sfcanet' in n.lower()]
    if len(sfca_results) < 2:
        return

    names = [n for n, _ in sfca_results]
    fns = [m['FN'] for _, m in sfca_results]
    recalls = [m['Recall'] for _, m in sfca_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # False Negatives
    bars1 = ax1.bar(names, fns, color=['#FF5722', '#4CAF50'], edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('False Negatives (Missed Fakes)', fontsize=12)
    ax1.set_title('Punishment Effect on False Negatives', fontsize=13, fontweight='bold')
    for bar, val in zip(bars1, fns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', fontweight='bold', fontsize=12)

    # Recall
    bars2 = ax2.bar(names, recalls, color=['#FF5722', '#4CAF50'], edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Recall (%)', fontsize=12)
    ax2.set_title('Punishment Effect on Recall', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars2, recalls):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Punishment comparison saved: {save_path}")


def main():
    print("=" * 70)
    print("  PHASE 2 - EVALUATION & PUNISHMENT ANALYSIS")
    print("=" * 70)

    # --- Define models to evaluate --------------------------------------
    eval_configs = [
        ('DCT-CNN Standalone (pw=1)', 'dct_cnn', 'dct_cnn_standalone_pw1_best.pth', 'dct_only', True),
        ('SFCANet-v2 (pw=1)', 'sfcanet_v2', 'sfcanet_v2_pw1_best.pth', 'sfcanet', False),
        ('SFCANet-v2 (pw=5)', 'sfcanet_v2', 'sfcanet_v2_pw5_best.pth', 'sfcanet', False),
        ('Xception (pw=1)', 'xception', 'xception_pw1_best.pth', 'sfcanet', False),
        ('Xception (pw=5)', 'xception', 'xception_pw5_best.pth', 'sfcanet', False),
        ('ConvNeXt (pw=1)', 'convnext_tiny', 'convnext_tiny_pw1_best.pth', 'sfcanet', False),
        ('ConvNeXt (pw=5)', 'convnext_tiny', 'convnext_tiny_pw5_best.pth', 'sfcanet', False),
        ('Swin-T (pw=1)', 'swin_tiny_patch4_window7_224', 'swin_tiny_patch4_window7_224_pw1_best.pth', 'sfcanet', False),
        ('Swin-T (pw=5)', 'swin_tiny_patch4_window7_224', 'swin_tiny_patch4_window7_224_pw5_best.pth', 'sfcanet', False),
    ]

    all_results = []
    roc_data = []

    for display_name, model_type, ckpt_name, mode, use_balanced in eval_configs:
        print(f"\n{'-'*50}")
        print(f"  {display_name}")
        print(f"{'-'*50}")

        model = load_model(model_type, ckpt_name)
        if model is None:
            continue

        # Load appropriate test set
        _, _, test_ld = get_phase2_loaders(use_balanced=use_balanced)

        labels, probs, inf_ms = run_inference(model, test_ld, mode)
        metrics = compute_all_metrics(labels, probs)

        print(f"  Accuracy:  {metrics['Accuracy']:.2f}%")
        print(f"  Precision: {metrics['Precision']:.2f}%")
        print(f"  Recall:    {metrics['Recall']:.2f}%")
        print(f"  F1-Score:  {metrics['F1-Score']:.2f}%")
        print(f"  AUC-ROC:   {metrics['AUC-ROC']:.4f}")
        print(f"  TP={metrics['TP']} FP={metrics['FP']} "
              f"TN={metrics['TN']} FN={metrics['FN']}")
        print(f"  Inference: {inf_ms:.2f} ms/image")

        all_results.append((display_name, metrics, inf_ms))

        # ROC
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_data.append((display_name, fpr, tpr, metrics['AUC-ROC']))

        # Confusion matrix
        cm_path = os.path.join(PHASE2_RESULTS_DIR,
                               f"{ckpt_name.replace('_best.pth', '')}_confusion_matrix.png")
        plot_confusion_matrix(labels, probs, display_name, cm_path)

        del model
        torch.cuda.empty_cache()

    if not all_results:
        print("\n[ERROR] No models evaluated. Run phase2_train.py first.")
        return

    # --- Comparison Table -----------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    print(header)
    print("-" * len(header))
    for name, m, _ in all_results:
        print(f"{name:<30} {m['Accuracy']:>6.2f}% {m['Precision']:>6.2f}% "
              f"{m['Recall']:>6.2f}% {m['F1-Score']:>6.2f}% {m['AUC-ROC']:>7.4f} "
              f"{m['TP']:>5} {m['FP']:>5} {m['TN']:>5} {m['FN']:>5}")

    # --- Save CSV -------------------------------------------------------
    csv_path = os.path.join(PHASE2_RESULTS_DIR, "phase2_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy(%)', 'Precision(%)', 'Recall(%)',
                          'F1-Score(%)', 'AUC-ROC', 'TP', 'FP', 'TN', 'FN',
                          'Inference(ms)'])
        for name, m, inf in all_results:
            writer.writerow([name, f"{m['Accuracy']:.2f}", f"{m['Precision']:.2f}",
                             f"{m['Recall']:.2f}", f"{m['F1-Score']:.2f}",
                             f"{m['AUC-ROC']:.4f}", m['TP'], m['FP'], m['TN'], m['FN'],
                             f"{inf:.2f}"])
    print(f"\n[CSV] Saved: {csv_path}")

    # --- ROC Curves -----------------------------------------------------
    roc_path = os.path.join(PHASE2_RESULTS_DIR, "phase2_roc_curves.png")
    filtered_roc_data = [d for d in roc_data if "DCT-CNN Standalone" not in d[0]]
    plot_roc_curves(filtered_roc_data, roc_path)

    # --- Punishment Comparison ------------------------------------------
    punish_path = os.path.join(PHASE2_RESULTS_DIR, "punishment_comparison.png")
    plot_punishment_comparison(all_results, punish_path)

    print(f"\n{'='*70}")
    print(f"  [OK] PHASE 2 EVALUATION COMPLETE")
    print(f"    Results: {PHASE2_RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
