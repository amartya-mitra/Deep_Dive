"""
Yin-Yang Sanity Check Experiment

Trains MLPs of varying depths on the Yin-Yang 2D dataset.
Captures: Loss/Accuracy curves, Inter-Layer CKA heatmaps, Test Accuracy vs Depth.
No latent factor analysis (no CKA with latents, no linear probes).
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.show = lambda: None

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
root_dir = os.path.join(parent_dir, '..', '..')
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'utils'))

from utils.misc import train_model, inter_layer_cka_analysis
from utils.toy_model import Classifier
from data.toy.yinyang.yinyang_data import generate_yinyang


def run_experiment():
    print("Starting Yin-Yang Sanity Check Experiment...", flush=True)
    
    # 1. Generate Data
    print("Generating Yin-Yang dataset...", flush=True)
    X_all, y_all = generate_yinyang(n_samples=10000, seed=42)
    
    # Train/Test split (80/20)
    n_total = len(X_all)
    n_train = int(0.8 * n_total)
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}", flush=True)
    
    # Class balance
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    print(f"Class Balance: Pos={n_pos} ({n_pos/len(y_train):.2%}), Neg={n_neg} ({n_neg/len(y_train):.2%})", flush=True)
    
    # Save a scatter plot of the dataset
    figs_dir = os.path.join(root_dir, 'figs/yinyang')
    os.makedirs(figs_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 6))
    colors = ['#1a1a6e', '#f0c040']  # Blue-ish, Yellow-ish to match ref
    for c in [0, 1]:
        mask = y_all == c
        plt.scatter(X_all[mask, 0].numpy(), X_all[mask, 1].numpy(),
                    c=colors[c], s=2, alpha=0.6, label=f'Class {c}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Yin-Yang Dataset')
    plt.legend(markerscale=5)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'dataset.png'), dpi=200)
    plt.close()
    print("Dataset plot saved.", flush=True)
    
    # 2. Config
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Running on device: {device}", flush=True)
    
    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_dim = 64
    input_dim = 2
    num_classes = 2
    
    test_accuracies = []
    
    # 3. Training Loop
    for depth in depths:
        print(f"\nTraining Depth {depth}...", flush=True)
        
        model = Classifier(input_dim, depth, hidden_dim, num_classes, 'relu')
        
        train_dict = {
            'epochs': 500,
            'lr': 0.01,
            'momentum': 0.9,
            'optimizer': 'sgd',
            'min_loss_change': 0.0001,
            'no_improve_threshold': 50,
            'loss_mp': 1,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'wandb': False
        }
        
        # Train
        model = train_model(model, train_dict['epochs'], True, True,
                            train_dict, X_train, y_train, seed=42,
                            save_path=os.path.join(figs_dir, f'metrics_depth_{depth}.png'))
        
        # Evaluate
        with torch.no_grad():
            model.eval()
            out_test = model(X_test.to(device))
            acc_test = (out_test.argmax(1) == y_test.to(device)).float().mean().item()
            test_accuracies.append(acc_test)
        
        print(f"Depth {depth}: Test Acc = {acc_test:.4f}", flush=True)
        
        # Inter-Layer CKA
        inter_layer_cka_analysis(model, X_test,
                                 True,
                                 save_path=os.path.join(figs_dir, f'inter_layer_cka_depth_{depth}.png'))
    
    # 4. Summary Plot: Test Accuracy vs Depth
    print("\nGenerating summary plots...", flush=True)
    plt.figure()
    plt.plot(depths, test_accuracies, label='Test Accuracy', marker='o', color='blue')
    plt.xlabel('Model Depth')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Model Depth (Yin-Yang)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figs_dir, 'test_acc_vs_depth.png'))
    plt.close()
    
    print("Experiment Complete.", flush=True)


if __name__ == '__main__':
    run_experiment()
