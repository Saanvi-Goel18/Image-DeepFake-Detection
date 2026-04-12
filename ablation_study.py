"""
Ablation Study Evaluation
=========================
Compares standalone ResNet-50 (Phase 1), standalone DCT-CNN, and SFCANet-v2.
Forces CPU execution to avoid out-of-memory errors with the ongoing background training.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from phase2_dataset import get_phase2_loaders
from dct_cnn import DCTCNN
from sfcanet import SFCANet
from phase2_config import PHASE2_RESULTS_DIR

# Force CPU to perfectly protect the background training run
DEVICE = torch.device('cpu')

def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        'Accuracy': accuracy_score(labels, preds) * 100,
        'AUC-ROC': roc_auc_score(labels, probs),
        'Recall': recall_score(labels, preds) * 100,
        'F1-Score': f1_score(labels, preds) * 100
    }

def load_resnet50():
    print("[Ablation] Loading ResNet-50 (Phase 1)...")
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    ckpt_path = "phase1_checkpoints/resnet50_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        # Handle if the state dict didn't wrap the model, etc.
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    return model.to(DEVICE).eval()

def load_dct_cnn():
    print("[Ablation] Loading DCT-CNN (Phase 2)...")
    model = DCTCNN(standalone=True)
    ckpt_path = "phase2_checkpoints/dct_cnn_standalone_pw1_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(DEVICE).eval()

def load_sfcanet():
    print("[Ablation] Loading SFCANet-v2 (Phase 2)...")
    # v2 has layer4 unfrozen
    model = SFCANet(resnet50_ckpt_path="phase1_checkpoints/resnet50_best.pth", use_gradient_checkpoint=False, unfreeze_layer4=True)
    ckpt_path = "phase2_checkpoints/sfcanet_v2_pw1_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(DEVICE).eval()

def run_inference(model, loader, model_type):
    all_probs = []
    all_labels = []
    
    # We only take the first 10 batches (160 images) to make standard CPU inference fast 
    # and provide lightning-fast updates without blocking for an hour.
    for i, batch in enumerate(loader):
        # We process the entire dataset now
        images, dct_coeffs, labels = batch
        images = images.to(DEVICE)
        dct_coeffs = dct_coeffs.to(DEVICE)
        
        with torch.no_grad():
            if model_type == 'resnet':
                out = model(images)
            elif model_type == 'dct':
                out = model(dct_coeffs)
            elif model_type == 'sfcanet':
                out = model(images, dct_coeffs)
                
        probs = torch.sigmoid(out).cpu().squeeze().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_labels.extend(labels.numpy().tolist())
        
    import numpy as np
    return np.array(all_labels), np.array(all_probs)

def main():
    print("=" * 60)
    print("   ABLATION STUDY (FAST CPU EVALUATION)")
    print("=" * 60)
    
    _, _, test_ld = get_phase2_loaders(use_balanced=False)
    
    models_to_test = [
        ('ResNet-50 (Spatial Only)', load_resnet50, 'resnet'),
        ('DCT-CNN (Frequency Only)', load_dct_cnn, 'dct'),
        ('SFCANet-v2 (Fusion)', load_sfcanet, 'sfcanet')
    ]
    
    results = []
    roc_data = []

    for name, loader_fn, mtype in models_to_test:
        model = loader_fn()
        labels, probs = run_inference(model, test_ld, mtype)
        metrics = compute_metrics(labels, probs)
        results.append((name, metrics))
        
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_data.append((name, fpr, tpr, metrics['AUC-ROC']))
        del model
        
    # Table
    print("\n" + "=" * 60)
    header = f"{'Model':<25} {'Acc':>7} {'Recall':>7} {'AUC':>7}"
    print(header)
    print("-" * len(header))
    for name, m in results:
        print(f"{name:<25} {m['Accuracy']:>6.2f}% {m['Recall']:>6.2f}% {m['AUC-ROC']:>7.4f}")
    
    # ROC Plot
    plt.figure(figsize=(7, 5))
    for name, fpr, tpr, auc in roc_data:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", lw=2)
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.title("Ablation Study ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(PHASE2_RESULTS_DIR, "ablation_roc.png")
    plt.savefig(save_path)
    print(f"\n[Saved Plot] {save_path}")

    # CSV Export
    import csv
    csv_path = os.path.join(PHASE2_RESULTS_DIR, "ablation_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Recall', 'AUC'])
        for name, m in results:
            writer.writerow([name, f"{m['Accuracy']:.2f}", f"{m['Recall']:.2f}", f"{m['AUC-ROC']:.4f}"])
    print(f"[Saved CSV] {csv_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
