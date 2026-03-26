# Handover: dSprites Latent Recovery Experiment Execution

**Date**: 2026-02-11
**Objective**: Execute an experiment to test if neural network extractor layers recover true latent factors (Core) vs spurious correlations (Position) using the dSprites dataset.

## Context
This conversation focused on the **implementation and execution** phase of the experiment. The initial setup and planning were established in a previous session (referenced as `Handover_dSprites_Latent_Recovery.md`).

## Accomplishments

### 1. Data Pipeline Implementation
- Created `data/toy/dsprites/dsprites_data.py`:
    - `load_dsprites`: Loads and subsamples the dataset.
    - `generate_labels`: Implements the "Combined Feature" logic.
        - **Core Label**: Logical OR of Shape (Square vs Ellipse/Heart) and Scale (> Median).
        - **Spurious correlation**: Handled via dataset filtering in the training script to ensure $P(y = \text{Spurious}) = 0.9$.

### 2. Analysis Tools
- Modified `utils/misc.py`:
    - Added `latent_CKA_analysis`: Computes CKA similarity between layer activations and Ground Truth Latents.
    - Added `linear_probe_analysis`: Trains Logistic Regression probes to predict latents from layer activations.
    - Updated `train_model` to support saving loss/error evolution plots.
- Modified `utils/plot.py`:
    - Added `plot_latent_cka_comparison`.
    - Updated `plot_metrics` to save plots to disk (non-interactive backend).

### 3. Experiment Execution (`train_dsprites.py`)
- Created a robust training script that:
    - Sweeps over model depths (1 to 8).
    - Constructs a **Biased Training Set** ($p=0.9$ correlation between Core and Spurious).
    - Constructs **OOD Test Sets** ($p=0.5$ random correlation).
    - Performs full training, CKA analysis, Linear Probe analysis, and Metrics plotting.
- **Outcome**: Successfully ran for all 8 depths.

## Key Findings
- **Additive Mechanism**: Models consistently achieved ~90% ID accuracy and ~75% OOD accuracy. This suggests the network learns a logical **OR** of both features ($P(\text{Correct}) \approx 0.5 + 0.5 \times P(\text{Core Matches})$?), rather than "tunneling" on just one.
- **No Extractor-Tunnel Boundary**: CKA analysis showed that Spurious feature information (Position) is maintained deep into the network, often co-existing with Core feature information.
- **Optimization**: Deeper models (7-8) showed higher training error but slightly better OOD generalization (~83%), possibly due to implicit regularization or harder optimization filtering out "easy" spurious paths.

## Artifacts & Locations
- **Code**:
    - `data/toy/dsprites/dsprites_data.py`
    - `data/toy/dsprites/train_dsprites.py`
    - `utils/misc.py` (Modified)
    - `utils/plot.py` (Modified)
- **Results/Figures**:
    - `figs/dsprites/`: Contains `cka_depth_*.png`, `metrics_depth_*.png`, `probe_depth_*.png`, `id_vs_ood_acc.png`.
- **Documentation**:
    - `walkthrough.md`: Detailed walkthrough of results with embedded plots (in artifacts directory).

## Next Steps for Future Agent
- **Investigate Depth 7-8**: Why did optimization fail (high loss) but OOD improve?
- **Feature Ablation**: Try to explicitly ablate the Spurious feature input (mask X) at test time to measure Core-only performance.
- **Intervention**: Can we use the CKA peaks to "guide" the model to discard Spurious information earlier?
