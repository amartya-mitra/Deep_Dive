import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
# Disable plt.show() to prevent blocking
plt.show = lambda: None

from collections import defaultdict
import datetime

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # data/toy
root_dir = os.path.join(parent_dir, '..', '..') # Deep_Dive
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'utils'))

from utils.lib import *
from utils.toy_model import *
from utils.misc import *
from utils.plot import *
from data.toy.dsprites.dsprites_data import load_dsprites, generate_labels

def filter_dataset_for_bias(X, latents_classes, latents_values, core_indices, spurious_indices, p_correlation, seed=42):
    """
    Filter dataset to enforce spurious correlation.
    We want Y (determined by Core) to correlate with Spurious Feature with probability `p_correlation`.
    Since Core and Spurious are naturally independent in dSprites, we must undersample the 'misaligned' examples.
    """
    np.random.seed(seed)
    
    # 1. Determine Core Labels (Target Y)
    # Using the logic from generate_labels but just getting the core part
    # We call generate_labels but we only care about core_labels
    _, core_labels = generate_labels(latents_classes, latents_values, core_indices, spurious_indices, 1.0, seed)
    y = core_labels # This is the TRUE label
    
    # 2. Determine Spurious Feature 'Bit'
    # PosX/PosY > Median -> 1, else 0.
    pos_x = latents_values[:, spurious_indices[0]]
    pos_y = latents_values[:, spurious_indices[1]]
    spurious_bit = ((pos_x > pos_x.median()).long() + (pos_y > pos_y.median()).long() >= 1).long()
    
    # 3. Align Check
    aligned = (y == spurious_bit)
    
    # 4. Filter
    # We want P(aligned) = p_correlation
    # Currently P(aligned) ~ 0.5 (random)
    # Let N_aligned be count of aligned. N_misaligned be count of misaligned.
    # We want N_aligned / (N_aligned + N_misaligned_keep) = p
    # N_aligned = p * N_aligned + p * N_misaligned_keep
    # N_aligned * (1-p) = p * N_misaligned_keep
    # N_misaligned_keep = N_aligned * (1-p)/p
    
    indices = np.arange(len(y))
    aligned_indices = indices[aligned]
    misaligned_indices = indices[~aligned]
    
    n_aligned = len(aligned_indices)
    n_misaligned = len(misaligned_indices)
    
    if p_correlation >= 0.5:
        # We need to drop misaligned
        n_keep_mis = int(n_aligned * (1 - p_correlation) / p_correlation)
        if n_keep_mis < n_misaligned:
            keep_mis_indices = np.random.choice(misaligned_indices, n_keep_mis, replace=False)
            final_indices = np.concatenate([aligned_indices, keep_mis_indices])
        else:
            # We don't have enough aligned samples to drive p up to target? 
            # Or we have too few misaligned? 
            # If n_keep_mis >= n_misaligned, we keep all misaligned, and we might need to drop aligned?
            # No, if we want high correlation, we primarily drop misaligned.
            # If we keep all misaligned, p = n_aligned / (n_aligned + n_misaligned) < target.
            # So we must drop misaligned.
            # The calculation holds.
            final_indices = np.concatenate([aligned_indices, misaligned_indices]) # Can't achieve target, keep all
    else:
        # Want negative correlation?
        # Similar logic.
        final_indices = indices
        
    np.random.shuffle(final_indices)
    
    return X[final_indices], y[final_indices], latents_classes[final_indices], latents_values[final_indices]
    

def run_experiment():
    print("Starting dSprites Latent Recovery Experiment...", flush=True)
    
    # 1. Setup Data
    dsprites_path = os.path.join(parent_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    # Load more samples to allow for filtering
    X_raw, _, lv_raw, _, lc_raw, _ = load_dsprites(dsprites_path, n_samples=50000)
    
    # Concatenate back to split manually after filtering
    # Actually load_dsprites splits 80/20. Let's just use the raw returns if we can, or combine.
    # load_dsprites returns tensors.
    
    # Re-combine for filtering
    # note: load_dsprites does a split. I'll just load again or combine.
    # The function returns X_train, X_test...
    # I'll combine them.
    # But wait, I can just not use the split and assume `X_raw` (train) is typically large enough.
    
    # Filter for Training Set (High Bias)
    print("Constructing Biased Training Set (p=0.9)...")
    X_train, y_train, lc_train, lv_train = filter_dataset_for_bias(
        X_raw, lc_raw, lv_raw, [1, 2], [4, 5], 0.9
    )
    print(f"Training Set Size: {len(X_train)}", flush=True)
    
    # Check Class Balance
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    print(f"Training Class Balance: Pos={n_pos} ({n_pos/len(y_train):.2%}), Neg={n_neg} ({n_neg/len(y_train):.2%})", flush=True)
    
    # Construct OOD Test Set (No Bias, p=0.5)
    # Use the 'Test' part from load_dsprites (which comes from the remaining pool)
    # But load_dsprites returns split data from n_samples.
    # To get distinct OOD data, we should reload or use the other split.
    # Let's use the 'Test' returned by load_dsprites as the pool for OOD.
    _, X_test_pool, _, lv_test_pool, _, lc_test_pool = load_dsprites(dsprites_path, n_samples=5000, seed=999) # Different seed
    
    print("Constructing OOD Test Set (p=0.5)...")
    # p=0.5 means do nothing (natural distribution), but let's filter to be sure or just take it.
    # Natural distribution is independent, so p=0.5.
    # However, to compare accuracy, we might want to balance class 0 and 1.
    X_ood, y_ood, lc_ood, lv_ood = filter_dataset_for_bias(
        X_test_pool, lc_test_pool, lv_test_pool, [1, 2], [4, 5], 0.5
    )
    print(f"OOD Test Set Size: {len(X_ood)}")
    
    # ID Test Set (p=0.9 like train)
    X_id, y_id, lc_id, lv_id = filter_dataset_for_bias(
        X_test_pool, lc_test_pool, lv_test_pool, [1, 2], [4, 5], 0.9
    )
    print(f"ID Test Set Size: {len(X_id)}")
    
    # 2. Config
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Running on device: {device}", flush=True)
    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_dim = 120
    input_dim = 4096
    num_classes = 2
    
    # Metrics
    peak_cka_depths = []
    ood_accuracies = []
    id_accuracies = []
    
    figs_dir = os.path.join(root_dir, 'figs/dsprites')
    os.makedirs(figs_dir, exist_ok=True)
    
    # 3. Training Loop
    for depth in depths:
        print(f"\nTraining Depth {depth}...", flush=True)
        
        model = Classifier(input_dim, depth, hidden_dim, num_classes, 'relu')
        
        train_dict = {
            'epochs': 1500,
            'lr': 0.02,
            'momentum': 0.9,
            'optimizer': 'sgd',
            'min_loss_change': 0.0001,
            'no_improve_threshold': 50, # Reduced for speed
            'loss_mp': 1,
            'weight_decay': 1e-4, # L2 Regularization to prevent overfitting
            'batch_size': 64,     # Mini-batch training
            'wandb': False
        }
        
        # Train
        # Pass use_gpu=True generically, misc.py handles MPS/CUDA
        model = train_model(model, train_dict['epochs'], True, True, 
                            train_dict, X_train, y_train, seed=42,
                            save_path=os.path.join(figs_dir, f'metrics_depth_{depth}.png'))

        
        # Evaluate
        with torch.no_grad():
            model.eval()
            # ID Accuracy
            out_id = model(X_id.to(device))
            acc_id = (out_id.argmax(1) == y_id.to(device)).float().mean().item()
            id_accuracies.append(acc_id)
            
            # OOD Accuracy
            out_ood = model(X_ood.to(device))
            acc_ood = (out_ood.argmax(1) == y_ood.to(device)).float().mean().item()
            ood_accuracies.append(acc_ood)
            
        print(f"Depth {depth}: ID Acc = {acc_id:.4f}, OOD Acc = {acc_ood:.4f}", flush=True)
        
        # Analysis
        # Latent CKA
        # Define indices: Shape(1), Scale(2), PosX(4), PosY(5)
        # Note: In dSprites, Latents are: Color, Shape, Scale, Orientation, PosX, PosY
        # Shape(1), Scale(2) are CORE.
        # PosX(4), PosY(5) are SPURIOUS.
        # Orientation(3) is NOISE.
        
        latent_indices = {
            'All': [1, 2, 4, 5],
            'Core': [1, 2],
            'Spurious': [4, 5],
            'Noise': [3]
        }
        
        # We use the OOD set for Analysis (to see if it extracts features cleanly without bias)
        # Or ID? Usually we inspect representations on the validation set.
        # Let's use OOD set (balanced) to check representation quality.
        
        cka_results = latent_CKA_analysis(model, X_ood, lv_ood, latent_indices, 
                                          True, 
                                          save_path=os.path.join(figs_dir, f'cka_depth_{depth}.png'))
        
        # Find peak depth for Core
        core_cka = cka_results['Core']
        peak_layer = np.argmax(core_cka)
        peak_cka_depths.append(peak_layer)
        
        # Linear Probe
        # Probe using classes (integers)
        probe_indices = {
            'Shape': 1,
            'Scale': 2,
            'PosX': 4,
            'PosY': 5
        }
        linear_probe_analysis(model, X_ood, lc_ood, probe_indices, 
                              torch.cuda.is_available(),
                              save_path=os.path.join(figs_dir, f'probe_depth_{depth}.png'))


    # 4. Summary Plots
    print("\ngenerating summary plots...")
    
    # ID vs OOD Accuracy
    plt.figure()
    plt.plot(depths, id_accuracies, label='ID Accuracy', marker='o')
    plt.plot(depths, ood_accuracies, label='OOD Accuracy', marker='s')
    plt.xlabel('Model Depth')
    plt.ylabel('Accuracy')
    plt.title('ID vs OOD Accuracy across Depths')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figs_dir, 'id_vs_ood_acc.png'))
    
    # Peak CKA Layer vs Model Depth
    plt.figure()
    plt.plot(depths, peak_cka_depths, marker='o', label='Core CKA Peak')
    plt.plot(depths, depths, linestyle='--', color='gray', label='Output Layer (Identity)')
    plt.plot(depths, [d/2 for d in depths], linestyle=':', color='gray', label='Middle')
    plt.xlabel('Model Depth')
    plt.ylabel('Layer Index')
    plt.title('Location of Extractor-Tunnel Boundary')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figs_dir, 'peak_layer_trend.png'))
    
    print("Experiment Complete.")

if __name__ == "__main__":
    run_experiment()
