"""
Cross-Dataset Validation: FaceForensics++ Frame Evaluation
===========================================================
This script takes the FaceForensics++ dataset (which was processed 
into segments for the Video project) and unpacks the stacked frames 
so they can be evaluated by our spatial-frequency Image models (SFCANet-v2).
This will mathematically prove generalizability and resistance to overfitting.
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import Image Project configs and models
from phase2_config import (
    DEVICE, PHASE2_CHECKPOINT_DIR, PHASE2_RESULTS_DIR, PHASE1_RESNET50_CKPT,
    IMAGENET_MEAN, IMAGENET_STD, DCT_BLOCK_SIZE
)
from sfcanet import SFCANet
import cv2

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFPP_CROPS_DIR = os.path.join(os.path.dirname(BASE_DIR), "Visual Images DeepFake Detection", "data_crops", "test")

# Re-use the exact same block DCT logic from Image project
def compute_block_dct(img_np: np.ndarray, block_size: int = 8) -> np.ndarray:
    H, W, C = img_np.shape
    bh = H // block_size
    bw = W // block_size
    dct_out = np.zeros((C * block_size * block_size, bh, bw), dtype=np.float32)
    for c in range(C):
        ch = img_np[:, :, c].astype(np.float32) - 128.0
        for i in range(bh):
            for j in range(bw):
                block = ch[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                dct_out[c*block_size*block_size:(c+1)*block_size*block_size, i, j] = cv2.dct(block).flatten()
    dct_out = np.sign(dct_out) * np.log1p(np.abs(dct_out))
    return dct_out

_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class FFPPImageDataset(Dataset):
    """
    Unpacks FaceForensics++ segment .npy files into individual image frames
    so they can be evaluated by the 2D image models.
    """
    def __init__(self, crops_dir):
        super().__init__()
        self.crops_dir = crops_dir
        self.frames = []
        
        print(f"[Dataset] Scanning FFPP Crops in: {crops_dir}")
        for cls_dir in os.listdir(crops_dir):
            c_path = os.path.join(crops_dir, cls_dir)
            if not os.path.isdir(c_path): continue
            
            label = 0 if cls_dir == "original" else 1
            vid_dirs = os.listdir(c_path)
            
            for vid_id in vid_dirs:
                v_path = os.path.join(c_path, vid_id)
                if not os.path.isdir(v_path): continue
                
                for f in os.listdir(v_path):
                    if f.endswith('.npy'):
                        npy_path = os.path.join(v_path, f)
                        # We don't load the array yet to save RAM, just record the path
                        # and the fact that we will extract 8 frames from it.
                        for frame_idx in range(8):  
                            self.frames.append((npy_path, frame_idx, label))
                            
        print(f"[Dataset] Found {len(self.frames)} individual test frames across FFPP.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        npy_path, frame_idx, label = self.frames[idx]
        
        try:
            # We have to load the .npy file and pick out the specific frame
            stacked_frames = np.load(npy_path)
            frame_np = stacked_frames[frame_idx]
        except Exception as e:
            # Fallback black frame if corruption occurs
            frame_np = np.zeros((224, 224, 3), dtype=np.uint8)
            
        rgb = _val_transform(frame_np)
        dct = torch.from_numpy(compute_block_dct(frame_np, DCT_BLOCK_SIZE))
        
        return rgb, dct, torch.tensor(label, dtype=torch.float32)

@torch.no_grad()
def run_cross_inference(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    # We'll batch process to speed it up
    for batch_idx, (images, dct_coeffs, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        dct_coeffs = dct_coeffs.to(DEVICE, non_blocking=True)
        
        with autocast():
            outputs = model(images, dct_coeffs)
            
        probs = torch.sigmoid(outputs).cpu().squeeze().numpy()
        if probs.ndim == 0:
            all_probs.append(probs.item())
        else:
            all_probs.extend(probs.tolist())
            
        all_labels.extend(labels.numpy().tolist())
        
        if batch_idx % 100 == 0:
            print(f"  Processed batched {batch_idx}/{len(loader)}...")
            
    return np.array(all_labels), np.array(all_probs)

def main():
    print("=" * 70)
    print("  CROSS-DATASET VALIDATION (IMAGE MODELS -> FACEFORENSICS++)")
    print("=" * 70)
    
    if not os.path.exists(FFPP_CROPS_DIR):
        print(f"[ERROR] FaceForensics++ crops not found at {FFPP_CROPS_DIR}")
        return
        
    ds = FFPPImageDataset(FFPP_CROPS_DIR)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load SFCANet-v2 pw5 (which had the highest recall on original validation)
    ckpt_name = "sfcanet_v2_pw5_best.pth"
    path = os.path.join(PHASE2_CHECKPOINT_DIR, ckpt_name)
    
    print(f"\n[Model] Loading {ckpt_name}...")
    rckpt = PHASE1_RESNET50_CKPT if os.path.exists(PHASE1_RESNET50_CKPT) else None
    model = SFCANet(resnet50_ckpt_path=rckpt, use_gradient_checkpoint=False, unfreeze_layer4=True)
    
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(DEVICE).eval()
    
    print("\n[Inference] Running large-scale unseen validation...")
    t0 = time.time()
    labels, probs = run_cross_inference(model, loader)
    elapsed = time.time() - t0
    print(f"[Inference] Completed in {elapsed/60:.1f} minutes.")
    
    # Metrics
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, zero_division=0) * 100
    rec = recall_score(labels, preds, zero_division=0) * 100
    f1 = f1_score(labels, preds, zero_division=0) * 100
    auc = roc_auc_score(labels, probs)
    
    print("\n" + "=" * 50)
    print("  FACEFORENSICS++ GENERALIZATION METRICS")
    print("=" * 50)
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall   : {rec:.2f}%")
    print(f"  F1-Score : {f1:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("=" * 50)
    
    # Output visualizations
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Cross-Validation FF++ CM', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(PHASE2_RESULTS_DIR, "cross_val_ffpp_cm.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"\n[OK] Confusion matrix exported to: {cm_path}")

if __name__ == "__main__":
    main()
