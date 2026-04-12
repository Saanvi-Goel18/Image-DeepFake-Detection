"""
Step 5: Evaluation & Comparison Table
=======================================
1. Load best checkpoints for all 3 models
2. Evaluate on test set
3. Compute: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Inference Time
4. Generate: Comparison table (terminal + CSV + LaTeX), ROC curves, confusion matrices

Usage:
    python 05_evaluate.py
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR, MODEL_NAMES, PHYSICAL_BATCH_SIZE
from dataset import get_dataloaders
from models import get_model


def load_best_model(model_name):
    """Load the best checkpoint for a given model."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Evaluate] WARNING: No checkpoint found for {model_name}: {ckpt_path}")
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = get_model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"[Evaluate] Loaded {model_name} (epoch {checkpoint['epoch']}, "
          f"val_auc={checkpoint['best_auc']:.4f})")
    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, test_loader):
    """Run inference on test set and return all predictions + timing."""
    model.eval()
    all_probs = []
    all_labels = []
    total_time = 0.0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        batch_size = images.size(0)

        # Time inference
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast():
            outputs = model(images)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time += (end - start)
        total_samples += batch_size

        probs = torch.sigmoid(outputs).cpu().squeeze().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_labels.extend(labels.numpy().tolist())

    avg_inference_ms = (total_time / total_samples) * 1000  # ms per image
    return np.array(all_labels), np.array(all_probs), avg_inference_ms


def compute_metrics(labels, probs, threshold=0.5):
    """Compute all classification metrics."""
    preds = (probs >= threshold).astype(int)

    return {
        'Accuracy':  accuracy_score(labels, preds) * 100,
        'Precision': precision_score(labels, preds, zero_division=0) * 100,
        'Recall':    recall_score(labels, preds, zero_division=0) * 100,
        'F1-Score':  f1_score(labels, preds, zero_division=0) * 100,
        'AUC-ROC':   roc_auc_score(labels, probs),
    }


def plot_roc_curves(roc_data, save_path):
    """Plot overlaid ROC curves for all models."""
    plt.figure(figsize=(8, 6))

    colors = ['#2196F3', '#FF5722', '#4CAF50']
    for (name, fpr, tpr, auc_val), color in zip(roc_data, colors):
        display_name = name.replace('_', ' ').title()
        plt.plot(fpr, tpr, color=color, linewidth=2.0,
                 label=f'{display_name} (AUC = {auc_val:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Deepfake Detection Baselines', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] ROC curves saved: {save_path}")


def plot_confusion_matrix(labels, probs, model_name, save_path):
    """Plot confusion matrix heatmap."""
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    display_name = model_name.replace('_', ' ').title()
    plt.title(f'Confusion Matrix — {display_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_csv(results, save_path):
    """Save comparison table as CSV."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)',
            'F1-Score (%)', 'AUC-ROC', 'Inference Time (ms/image)'
        ])
        for name, metrics, inf_time in results:
            display_name = name.replace('_', ' ').title()
            writer.writerow([
                display_name,
                f"{metrics['Accuracy']:.2f}",
                f"{metrics['Precision']:.2f}",
                f"{metrics['Recall']:.2f}",
                f"{metrics['F1-Score']:.2f}",
                f"{metrics['AUC-ROC']:.4f}",
                f"{inf_time:.2f}"
            ])
    print(f"[CSV] Comparison table saved: {save_path}")


def generate_latex_table(results, save_path):
    """Generate LaTeX table for research paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Comparison of baseline deepfake detection models on the test set.}",
        r"    \label{tab:baseline_comparison}",
        r"    \begin{tabular}{lccccc}",
        r"        \toprule",
        r"        \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{F1-Score} & \textbf{AUC-ROC} \\",
        r"        \midrule",
    ]

    for name, metrics, _ in results:
        display_name = name.replace('_', r'\_').replace('mobilenetv3', 'MobileNetV3').replace(
            'large', 'Large').replace('efficientnet', 'EfficientNet-').replace(
            'b0', 'B0').replace('resnet50', 'ResNet-50')
        # Clean up display name
        if 'MobileNetV3' in display_name:
            display_name = 'MobileNetV3-Large'
        elif 'EfficientNet' in display_name:
            display_name = 'EfficientNet-B0'
        elif 'ResNet' in display_name:
            display_name = 'ResNet-50'

        lines.append(
            f"        {display_name} & "
            f"{metrics['Accuracy']:.2f}\\% & "
            f"{metrics['Precision']:.2f}\\% & "
            f"{metrics['Recall']:.2f}\\% & "
            f"{metrics['F1-Score']:.2f}\\% & "
            f"{metrics['AUC-ROC']:.4f} \\\\"
        )

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[LaTeX] Table saved: {save_path}")


def print_comparison_table(results):
    """Print a formatted comparison table to terminal."""
    try:
        from tabulate import tabulate
        headers = ['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)',
                    'F1-Score (%)', 'AUC-ROC', 'Inf. Time (ms)']
        rows = []
        for name, metrics, inf_time in results:
            display_name = {
                'mobilenetv3_large': 'MobileNetV3-Large',
                'efficientnet_b0': 'EfficientNet-B0',
                'resnet50': 'ResNet-50'
            }.get(name, name)
            rows.append([
                display_name,
                f"{metrics['Accuracy']:.2f}",
                f"{metrics['Precision']:.2f}",
                f"{metrics['Recall']:.2f}",
                f"{metrics['F1-Score']:.2f}",
                f"{metrics['AUC-ROC']:.4f}",
                f"{inf_time:.2f}"
            ])
        print("\n" + tabulate(rows, headers=headers, tablefmt='grid'))
    except ImportError:
        # Fallback: manual formatting
        print("\n" + "-" * 95)
        print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10} {'Inf.(ms)':>10}")
        print("-" * 95)
        for name, metrics, inf_time in results:
            display_name = {
                'mobilenetv3_large': 'MobileNetV3-Large',
                'efficientnet_b0': 'EfficientNet-B0',
                'resnet50': 'ResNet-50'
            }.get(name, name)
            print(f"{display_name:<20} "
                  f"{metrics['Accuracy']:>9.2f}% "
                  f"{metrics['Precision']:>9.2f}% "
                  f"{metrics['Recall']:>9.2f}% "
                  f"{metrics['F1-Score']:>9.2f}% "
                  f"{metrics['AUC-ROC']:>10.4f} "
                  f"{inf_time:>9.2f}")
        print("-" * 95)


def main():
    print("=" * 70)
    print("  DEEPFAKE DETECTION — EVALUATION & COMPARISON")
    print("=" * 70)

    # Load test data
    print("\n[Data] Loading test set...")
    _, _, test_loader = get_dataloaders()
    print(f"[Data] Test batches: {len(test_loader)} "
          f"(batch_size={PHYSICAL_BATCH_SIZE})")

    # Evaluate each model
    all_results = []
    roc_data = []

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*50}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*50}")

        model, checkpoint = load_best_model(model_name)
        if model is None:
            continue

        # Run inference
        labels, probs, inf_time_ms = evaluate_model(model, test_loader)
        print(f"  Inference time: {inf_time_ms:.2f} ms/image")

        # Compute metrics
        metrics = compute_metrics(labels, probs)
        for k, v in metrics.items():
            if k == 'AUC-ROC':
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:.2f}%")

        all_results.append((model_name, metrics, inf_time_ms))

        # ROC curve data
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_data.append((model_name, fpr, tpr, metrics['AUC-ROC']))

        # Confusion matrix
        cm_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(labels, probs, model_name, cm_path)

        # Cleanup
        del model
        torch.cuda.empty_cache()

    if not all_results:
        print("\n[ERROR] No models to evaluate. Run 04_train.py first.")
        return

    # ─── Generate Outputs ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print_comparison_table(all_results)

    # ROC curves
    roc_path = os.path.join(RESULTS_DIR, "roc_curves_comparison.png")
    plot_roc_curves(roc_data, roc_path)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    generate_comparison_csv(all_results, csv_path)

    # LaTeX
    latex_path = os.path.join(RESULTS_DIR, "comparison_table.tex")
    generate_latex_table(all_results, latex_path)

    print(f"\n{'='*70}")
    print(f"  ✓ EVALUATION COMPLETE")
    print(f"    Results directory: {RESULTS_DIR}")
    print(f"    CSV:    {csv_path}")
    print(f"    LaTeX:  {latex_path}")
    print(f"    ROC:    {roc_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
