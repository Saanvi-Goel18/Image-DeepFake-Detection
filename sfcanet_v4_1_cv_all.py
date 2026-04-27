"""
SFCANet-v4.1 Cross-Validation Master Runner
===========================================
Executes the 5-Fold CV loop for the new Laptop-Safe Subsampled Dataset.
"""

import time
from sfcanet_v4_1_cv import main as run_fold

def run_all_folds():
    print("=" * 70)
    print("  SFCANet-v4.1 5-Fold Cross-Validation Master Script")
    print("  Executing Fold 0 -> Fold 4 sequentially.")
    print("=" * 70)

    t0 = time.time()
    
    for fold_idx in range(5):
        try:
            print(f"\n\n{'*'*70}")
            print(f"*** INITIATING FOLD {fold_idx} ***")
            print(f"{'*'*70}\n")
            
            run_fold(fold_idx=fold_idx)
            
            print(f"*** FOLD {fold_idx} COMPLETED SUCCESSFULLY ***\n")
        except Exception as e:
            print(f"*** ERROR IN FOLD {fold_idx}: {e} ***")
            print("Stopping cross-validation loop.")
            break
            
    t1 = time.time()
    print("=" * 70)
    print(f"  All Folds Completed. Total Time: {(t1 - t0) / 3600:.2f} Hours.")
    print("=" * 70)

if __name__ == "__main__":
    run_all_folds()
