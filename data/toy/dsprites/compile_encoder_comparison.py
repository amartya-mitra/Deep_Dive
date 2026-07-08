"""
Compile a cross-encoder comparison for the STEP-1 null-result re-check.

Reads the per-depth numeric results saved by train_dsprites.py for each
ENCODER_MODE (the .npz files under logs/dsprites/<mode>/, which are the
authoritative numeric source behind each mode's summary plots) and produces:

  1. A CSV  : figs/dsprites/comparison/encoder_comparison.csv
       columns: encoder_mode, depth, id_acc, ood_acc,
                core_cka_peak_layer, rank_peak_layer, rank_peak_value
  2. A figure: figs/dsprites/comparison/encoder_comparison.png
       3 subplots (ID/OOD accuracy, core-CKA peak layer, rank-peak layer)
       vs depth, one line per encoder_mode.

Run after both modes' per-depth jobs have finished:
    python data/toy/dsprites/compile_encoder_comparison.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..', '..')  # Deep_Dive

ENCODER_MODES = ['flat_pixels', 'frozen_cnn']
ALL_DEPTHS = [3, 4, 5, 6, 7, 8]


def _rank_peak(rank_profile):
    """
    Locate an *interior* peak in a per-layer soft-rank profile.

    rank_profile[0] is the raw input and rank_profile[-1] the output layer;
    a peak at either endpoint is not an extractor-tunnel signature, so we
    report None for the layer in that case.

    Returns:
        (rank_peak_layer, rank_peak_value)
        rank_peak_layer : int index of interior peak, or None.
        rank_peak_value : float value at the interior peak, or None.
    """
    if rank_profile is None or len(rank_profile) == 0:
        return None, None
    idx = int(np.argmax(rank_profile))
    if idx == 0 or idx == len(rank_profile) - 1:
        return None, None
    return idx, float(rank_profile[idx])


def load_mode_rows(encoder_mode):
    """Read per-depth npz results for one encoder mode into a list of row dicts."""
    results_dir = os.path.join(root_dir, 'logs', 'dsprites', encoder_mode)
    rows = []
    for depth in ALL_DEPTHS:
        path = os.path.join(results_dir, f'results_depth_{depth}.npz')
        if not os.path.exists(path):
            print(f"[{encoder_mode}] missing depth {depth} at {path}; skipping.")
            continue
        data = np.load(path, allow_pickle=True)
        rank_profile = data['rank_profile'].tolist()
        rank_peak_layer, rank_peak_value = _rank_peak(rank_profile)
        rows.append({
            'encoder_mode':        encoder_mode,
            'depth':               depth,
            'id_acc':              float(data['id_acc']),
            'ood_acc':             float(data['ood_acc']),
            'core_cka_peak_layer': int(data['peak_cka']),
            'rank_peak_layer':     rank_peak_layer,
            'rank_peak_value':     rank_peak_value,
        })
    return rows


def main():
    all_rows = []
    for mode in ENCODER_MODES:
        all_rows.extend(load_mode_rows(mode))

    if not all_rows:
        print("No results found for any encoder mode. Run the per-depth jobs first.")
        return

    out_dir = os.path.join(root_dir, 'figs', 'dsprites', 'comparison')
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # CSV                                                                  #
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(out_dir, 'encoder_comparison.csv')
    fieldnames = ['encoder_mode', 'depth', 'id_acc', 'ood_acc',
                  'core_cka_peak_layer', 'rank_peak_layer', 'rank_peak_value']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mode in ENCODER_MODES:
        rows = [r for r in all_rows if r['encoder_mode'] == mode]
        if not rows:
            continue
        depths = [r['depth'] for r in rows]

        # Subplot 1: ID / OOD accuracy vs depth
        axes[0].plot(depths, [r['id_acc'] for r in rows],
                     marker='o', label=f'{mode} ID')
        axes[0].plot(depths, [r['ood_acc'] for r in rows],
                     marker='s', linestyle='--', label=f'{mode} OOD')

        # Subplot 2: core-CKA peak layer vs depth
        axes[1].plot(depths, [r['core_cka_peak_layer'] for r in rows],
                     marker='o', label=mode)

        # Subplot 3: rank-peak layer vs depth (None -> NaN so gaps show)
        rank_peaks = [r['rank_peak_layer'] if r['rank_peak_layer'] is not None
                      else np.nan for r in rows]
        axes[2].plot(depths, rank_peaks, marker='o', label=mode)

    axes[0].set_title('ID / OOD Accuracy vs Depth')
    axes[0].set_xlabel('Model Depth')
    axes[0].set_ylabel('Accuracy')

    axes[1].set_title('Core-CKA Peak Layer vs Depth')
    axes[1].set_xlabel('Model Depth')
    axes[1].set_ylabel('Peak Layer Index')
    # Reference: peak == depth means peak always at the output layer.
    dref = sorted({r['depth'] for r in all_rows})
    axes[1].plot(dref, dref, linestyle=':', color='gray', label='== depth (output)')

    axes[2].set_title('Interior Rank-Peak Layer vs Depth\n(gaps = no interior peak)')
    axes[2].set_xlabel('Model Depth')
    axes[2].set_ylabel('Rank-Peak Layer Index')

    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'encoder_comparison.png')
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
