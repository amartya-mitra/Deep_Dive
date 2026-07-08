import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
# Disable plt.show() to prevent blocking
plt.show = lambda: None

from collections import defaultdict

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # data/toy
root_dir = os.path.join(parent_dir, '..', '..') # Deep_Dive
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'utils'))

from utils.lib import *
from utils.toy_model import Classifier, CNNClassifier
from utils.misc import *
from utils.plot import *
from data.toy.dsprites.dsprites_data import (load_dsprites, generate_labels,
                                              generate_labels_6class,
                                              filter_dataset_for_bias_6class)


def downsample_flat(X_flat, from_size=64, to_size=32):
    """
    Downsample flattened square grayscale images via bilinear interpolation.

    Args:
        X_flat    : FloatTensor (N, from_size*from_size)
        from_size : int — side length of the input images.
        to_size   : int — side length of the downsampled images.
    Returns:
        FloatTensor (N, to_size*to_size)
    """
    X_img = X_flat.view(-1, 1, from_size, from_size)
    X_small = F.interpolate(X_img, size=(to_size, to_size),
                            mode='bilinear', align_corners=False)
    return X_small.view(-1, to_size * to_size)


def filter_dataset_for_bias(X, latents_classes, latents_values, core_indices, spurious_indices, p_correlation, seed=42, logic='OR'):
    """
    Filter dataset to enforce spurious correlation.
    We want Y (determined by Core) to correlate with Spurious Feature with probability `p_correlation`.
    Since Core and Spurious are naturally independent in dSprites, we must undersample the 'misaligned' examples.
    """
    np.random.seed(seed)

    # 1. Determine Core Labels (Target Y)
    _, core_labels = generate_labels(latents_classes, latents_values, core_indices, spurious_indices, 1.0, seed, logic=logic)
    y = core_labels # This is the TRUE label

    # 2. Determine Spurious Feature 'Bit'
    pos_x = latents_values[:, spurious_indices[0]]
    pos_y = latents_values[:, spurious_indices[1]]

    if logic == 'AND':
         spurious_bit = ((pos_x > pos_x.median()).long() + (pos_y > pos_y.median()).long() == 2).long()
    else:
         spurious_bit = ((pos_x > pos_x.median()).long() + (pos_y > pos_y.median()).long() >= 1).long()

    # 3. Align Check
    aligned = (y == spurious_bit)

    # 4. Filter
    indices = np.arange(len(y))
    aligned_indices = indices[aligned]
    misaligned_indices = indices[~aligned]

    n_aligned = len(aligned_indices)
    n_misaligned = len(misaligned_indices)

    if p_correlation >= 0.5:
        n_keep_mis = int(n_aligned * (1 - p_correlation) / p_correlation)
        if n_keep_mis < n_misaligned:
            keep_mis_indices = np.random.choice(misaligned_indices, n_keep_mis, replace=False)
            final_indices = np.concatenate([aligned_indices, keep_mis_indices])
        else:
            final_indices = np.concatenate([aligned_indices, misaligned_indices])
    else:
        n_keep_aln = int(n_misaligned * p_correlation / (1 - p_correlation))
        if n_keep_aln < n_aligned:
            keep_aln_indices = np.random.choice(aligned_indices, n_keep_aln, replace=False)
            final_indices = np.concatenate([keep_aln_indices, misaligned_indices])
        else:
            final_indices = np.concatenate([aligned_indices, misaligned_indices])

    np.random.shuffle(final_indices)

    if logic == 'AND':
        y_filtered = y[final_indices]
        pos_subset_indices = np.where(y_filtered == 1)[0]
        neg_subset_indices = np.where(y_filtered == 0)[0]

        n_pos = len(pos_subset_indices)
        n_neg = len(neg_subset_indices)

        if n_neg > n_pos:
            keep_neg_indices = np.random.choice(neg_subset_indices, n_pos, replace=False)
            balanced_indices = np.concatenate([pos_subset_indices, keep_neg_indices])
            final_indices = final_indices[balanced_indices]
            np.random.shuffle(final_indices)

    return X[final_indices], y[final_indices], latents_classes[final_indices], latents_values[final_indices]


def run_experiment(depth=None):
    """
    Run dSprites 6-class experiment.

    Args:
        depth : int or None — if given, trains only that depth; if None, trains all depths
                              sequentially and generates summary plots.
    """
    print("Starting dSprites 6-Class Latent Recovery Experiment...", flush=True)

    # ------------------------------------------------------------------ #
    # Config                                                               #
    # ------------------------------------------------------------------ #
    USE_SPURIOUS    = True   # Set False to disable spurious correlation
    P_CORRELATION   = 0.7   # Only used when USE_SPURIOUS=True

    # 'flat_pixels' -> Option A: no CNN, 32x32 downsampled flat input
    # 'frozen_cnn'  -> Option B: CNN encoder present but frozen (random
    #                  init, untrained), embed_dim=512
    ENCODER_MODE = os.environ.get('ENCODER_MODE', 'flat_pixels')
    assert ENCODER_MODE in ('flat_pixels', 'frozen_cnn')
    print(f"Running with ENCODER_MODE = {ENCODER_MODE}", flush=True)
    EMBED_DIM = 512          # frozen CNN encoder output dimension

    # ------------------------------------------------------------------ #
    # 1. Data                                                              #
    # ------------------------------------------------------------------ #
    dsprites_path = os.path.join(parent_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    # Training pool (all 3 shapes kept)
    X_raw, _, lv_raw, _, lc_raw, _ = load_dsprites(dsprites_path, n_samples=50000)
    if ENCODER_MODE == 'flat_pixels':
        X_raw = downsample_flat(X_raw)
    labels_raw, scale_bin_raw = generate_labels_6class(lc_raw, lv_raw)

    print("Constructing Training Set...", flush=True)
    X_train, y_train, lc_train, lv_train = filter_dataset_for_bias_6class(
        X_raw, labels_raw, scale_bin_raw, lc_raw, lv_raw,
        P_CORRELATION, use_spurious=USE_SPURIOUS
    )
    print(f"Training Set Size: {len(X_train)}", flush=True)
    for c in range(6):
        n = (y_train == c).sum().item()
        print(f"  Class {c}: {n} ({n/len(y_train):.1%})", flush=True)

    # Test pool (separate seed)
    _, X_test_pool, _, lv_test_pool, _, lc_test_pool = load_dsprites(
        dsprites_path, n_samples=5000, seed=999
    )
    if ENCODER_MODE == 'flat_pixels':
        X_test_pool = downsample_flat(X_test_pool)
    labels_test, scale_bin_test = generate_labels_6class(lc_test_pool, lv_test_pool)

    # OOD: no spurious correlation (natural distribution)
    print("Constructing OOD Test Set...", flush=True)
    X_ood, y_ood, lc_ood, lv_ood = filter_dataset_for_bias_6class(
        X_test_pool, labels_test, scale_bin_test, lc_test_pool, lv_test_pool,
        P_CORRELATION, use_spurious=False
    )
    print(f"OOD Test Set Size: {len(X_ood)}", flush=True)

    # ID: same spurious correlation as training
    print("Constructing ID Test Set...", flush=True)
    X_id, y_id, lc_id, lv_id = filter_dataset_for_bias_6class(
        X_test_pool, labels_test, scale_bin_test, lc_test_pool, lv_test_pool,
        P_CORRELATION, use_spurious=USE_SPURIOUS
    )
    print(f"ID Test Set Size: {len(X_id)}", flush=True)

    # Derived binary probe targets for OOD set
    scale_bin_ood    = (lv_ood[:, 2] > lv_ood[:, 2].median()).long()
    pos_x_ood        = lv_ood[:, 4]
    pos_y_ood        = lv_ood[:, 5]
    spurious_bit_ood = ((pos_x_ood > pos_x_ood.median()).long() +
                        (pos_y_ood > pos_y_ood.median()).long() == 2).long()
    lc_ood = torch.cat([lc_ood,
                        scale_bin_ood.unsqueeze(1),
                        spurious_bit_ood.unsqueeze(1)], dim=1)

    # Normalise using training statistics
    train_mean = X_train.mean(dim=0)
    train_std  = X_train.std(dim=0)
    train_std[train_std == 0] = 1.0
    X_train = (X_train - train_mean) / train_std
    X_ood   = (X_ood   - train_mean) / train_std
    X_id    = (X_id    - train_mean) / train_std

    # ------------------------------------------------------------------ #
    # 2. Config                                                            #
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Running on device: {device}", flush=True)

    all_depths  = [3, 4, 5, 6, 7, 8]
    depths_to_run = [depth] if depth is not None else all_depths
    hidden_dim  = 1024
    num_classes = 6
    input_dim   = 1024 if ENCODER_MODE == 'flat_pixels' else EMBED_DIM

    figs_dir = os.path.join(root_dir, 'figs/dsprites', ENCODER_MODE)
    dirs = {
        'cka':             os.path.join(figs_dir, 'cka'),
        'probe':           os.path.join(figs_dir, 'probe'),
        'inter_layer_cka': os.path.join(figs_dir, 'inter_layer_cka'),
        'metrics':         os.path.join(figs_dir, 'metrics'),
        'rank':            os.path.join(figs_dir, 'rank'),
        'summary':         os.path.join(figs_dir, 'summary'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    results_save_dir = os.path.join(root_dir, 'logs/dsprites', ENCODER_MODE)
    os.makedirs(results_save_dir, exist_ok=True)

    # Accumulators (used when running all depths sequentially)
    peak_cka_depths   = []
    ood_accuracies    = []
    id_accuracies     = []
    all_rank_profiles = {}

    # ------------------------------------------------------------------ #
    # 3. Training Loop                                                     #
    # ------------------------------------------------------------------ #
    for d in depths_to_run:
        print(f"\nTraining Depth {d}...", flush=True)

        if ENCODER_MODE == 'frozen_cnn':
            model = CNNClassifier(
                embed_dim=EMBED_DIM,
                num_hidden_layers=d,
                hidden_layer_width=hidden_dim,
                output_dim=num_classes,
                activation_func='relu',
                dropout=0.0,
                freeze_encoder=True,
            )
            n_trainable_enc = sum(p.requires_grad for p in model.encoder.parameters())
            print(f"  Frozen encoder sanity: trainable encoder params = "
                  f"{n_trainable_enc} (expected 0)", flush=True)
        else:  # flat_pixels
            model = Classifier(input_dim, d, hidden_dim, num_classes, 'relu', dropout=0.0)

        train_dict = {
            'epochs':              1500,
            'lr':                  0.005,
            'momentum':            0.9,
            'optimizer':           'sgd',
            'min_loss_change':     0.0001,
            'no_improve_threshold': 50,
            'loss_mp':             1,
            'weight_decay':        1e-4,
            'batch_size':          64,
            'label_smoothing':     0.0,
            'scheduler':           'cosine',
            'wandb':               False,
        }

        model = train_model(model, train_dict['epochs'], True, True,
                            train_dict, X_train, y_train, seed=42,
                            save_path=os.path.join(dirs['metrics'], f'metrics_depth_{d}.png'))

        # Evaluate
        with torch.no_grad():
            model.eval()
            out_id  = model(X_id.to(device))
            acc_id  = (out_id.argmax(1)  == y_id.to(device)).float().mean().item()
            out_ood = model(X_ood.to(device))
            acc_ood = (out_ood.argmax(1) == y_ood.to(device)).float().mean().item()
        id_accuracies.append(acc_id)
        ood_accuracies.append(acc_ood)
        print(f"Depth {d}: ID Acc = {acc_id:.4f}, OOD Acc = {acc_ood:.4f}", flush=True)

        # Latent CKA
        latent_indices = {
            'All':      [1, 2, 4, 5],
            'Core':     [1, 2],
            'Spurious': [4, 5],
            'Noise':    [3],
        }
        cka_results = latent_CKA_analysis(model, X_ood, lv_ood, latent_indices,
                                          True,
                                          save_path=os.path.join(dirs['cka'], f'cka_depth_{d}.png'))
        peak_cka = int(np.argmax(cka_results['Core']))
        peak_cka_depths.append(peak_cka)

        # Linear Probe
        probe_indices = {
            'Shape':        1,
            'Scale':        2,
            'PosX':         4,
            'PosY':         5,
            'Scale Bin':    6,
            'Spurious Bit': 7,
        }
        linear_probe_analysis(model, X_ood, lc_ood, probe_indices,
                              torch.cuda.is_available(),
                              save_path=os.path.join(dirs['probe'], f'probe_depth_{d}.png'))

        # Inter-Layer CKA
        inter_layer_cka_analysis(model, X_ood,
                                 torch.cuda.is_available(),
                                 save_path=os.path.join(dirs['inter_layer_cka'], f'inter_layer_cka_depth_{d}.png'))

        # Representation Rank
        rank_profile = layer_rep_rank_analysis(model, X_ood,
                                               torch.cuda.is_available(),
                                               save_path=os.path.join(dirs['rank'], f'rank_depth_{d}.png'))
        all_rank_profiles[d] = rank_profile

        # Save numeric results for this depth (used by --summary)
        np.savez(os.path.join(results_save_dir, f'results_depth_{d}.npz'),
                 id_acc=np.array(acc_id),
                 ood_acc=np.array(acc_ood),
                 peak_cka=np.array(peak_cka),
                 rank_profile=np.array(rank_profile))
        print(f"Depth {d} results saved.", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Summary Plots (sequential mode only)                             #
    # ------------------------------------------------------------------ #
    if depth is None:
        _generate_summary(all_depths, id_accuracies, ood_accuracies,
                          peak_cka_depths, all_rank_profiles, dirs['summary'])

    print("Experiment Complete.", flush=True)


def generate_summary_plots():
    """
    Read saved per-depth results from logs/dsprites/<ENCODER_MODE>/ and
    generate summary plots for that mode. Run after all parallel depth jobs
    for a given mode have finished:
        ENCODER_MODE=flat_pixels python data/toy/dsprites/train_dsprites.py --summary
    """
    encoder_mode = os.environ.get('ENCODER_MODE', 'flat_pixels')
    assert encoder_mode in ('flat_pixels', 'frozen_cnn')
    all_depths = [3, 4, 5, 6, 7, 8]
    results_save_dir = os.path.join(root_dir, 'logs/dsprites', encoder_mode)

    id_accuracies     = []
    ood_accuracies    = []
    peak_cka_depths   = []
    all_rank_profiles = {}
    depths_found      = []

    for d in all_depths:
        path = os.path.join(results_save_dir, f'results_depth_{d}.npz')
        if not os.path.exists(path):
            print(f"Warning: missing results for depth {d} at {path}. Skipping.", flush=True)
            continue
        data = np.load(path, allow_pickle=True)
        id_accuracies.append(float(data['id_acc']))
        ood_accuracies.append(float(data['ood_acc']))
        peak_cka_depths.append(int(data['peak_cka']))
        all_rank_profiles[d] = data['rank_profile'].tolist()
        depths_found.append(d)

    if not depths_found:
        print("No saved results found. Run per-depth jobs first.", flush=True)
        return

    summary_dir = os.path.join(root_dir, 'figs/dsprites', encoder_mode, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    _generate_summary(depths_found, id_accuracies, ood_accuracies,
                      peak_cka_depths, all_rank_profiles, summary_dir)


def _generate_summary(depths, id_accuracies, ood_accuracies,
                      peak_cka_depths, all_rank_profiles, summary_dir):
    """Generate summary plots from in-memory results."""
    print("\nGenerating summary plots...", flush=True)

    # Rank profiles overlaid across all depths
    fig, ax = plt.subplots(figsize=(9, 6))
    for d, ranks in all_rank_profiles.items():
        ax.plot(range(len(ranks)), ranks, marker='o', label=f'depth={d}')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Soft Rank (exp H)')
    ax.set_title('Representation Rank vs. Layer (all depths)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'rank_all_depths.png'))
    plt.close(fig)

    # ID vs OOD Accuracy
    fig, ax = plt.subplots()
    ax.plot(depths, id_accuracies, label='ID Accuracy', marker='o')
    ax.plot(depths, ood_accuracies, label='OOD Accuracy', marker='s')
    ax.set_xlabel('Model Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('ID vs OOD Accuracy across Depths')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'id_vs_ood_acc.png'))
    plt.close(fig)

    # Peak CKA Layer vs Model Depth
    fig, ax = plt.subplots()
    ax.plot(depths, peak_cka_depths, marker='o', label='Core CKA Peak')
    ax.plot(depths, depths, linestyle='--', color='gray', label='Output Layer (Identity)')
    ax.plot(depths, [d / 2 for d in depths], linestyle=':', color='gray', label='Middle')
    ax.set_xlabel('Model Depth')
    ax.set_ylabel('Layer Index')
    ax.set_title('Location of Extractor-Tunnel Boundary')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'peak_layer_trend.png'))
    plt.close(fig)

    print("Summary plots saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dSprites 6-class depth experiment")
    parser.add_argument('--depth', type=int, default=None,
                        help='Single depth to train (3–8). Omit to run all depths sequentially.')
    parser.add_argument('--summary', action='store_true',
                        help='Generate summary plots from saved per-depth results (run after all jobs finish).')
    args = parser.parse_args()

    if args.summary:
        generate_summary_plots()
    else:
        run_experiment(depth=args.depth)
