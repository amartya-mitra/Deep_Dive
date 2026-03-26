# Handover Plan: Latent Recovery Experiment on dSprites

## 1. Objective

Design and execute an experiment to test whether the **extractor layers** of a trained neural network recover the **true latent factors** underlying the dSprites dataset. This is the "Latent Recovery" hypothesis described in the project's research notes.

### Success Criteria
- Train MLPs of varying depths on dSprites with synthetic labels
- Demonstrate (or refute) that layer-wise CKA between hidden representations and ground-truth latents **peaks at intermediate depth**, corresponding to the extractor-tunnel boundary

---

## 2. Background & Context

### Project Context
This experiment is **Part 4** of the ongoing Deep Dive project. Previous parts established:
- **Part 1:** Rank/CKA analysis on core/spurious toy data → confirmed extractor-tunnel decomposition
- **Part 2:** NTK analysis on Yin-Yang → confirmed NTK rank peak at critical depth
- **Part 3:** Latent recovery on MNIST/CIFAR-10 (via DINO) → **inconclusive** (no strong correlation found)

Part 3 failed likely because: (a) the latent features were extracted via SSL (DINO), not ground truth, and (b) the latent-to-input mapping was nonlinear and unknown. **This experiment fixes both issues** by using dSprites, where ground-truth latents are known exactly.

### Existing Codebase Patterns
The codebase follows a consistent pattern (see `data/toy/latent/train_latent.py` as the reference template):
1. Import from `utils/` via relative path manipulation (`sys.path.append`)
2. Generate data → configure hyperparameters → train model → analyze
3. Analysis uses: `compute_layer_rank()`, `layerwise_CKA()`, `get_all_layer_outputs()`
4. All utility code lives in `utils/` (`data.py`, `toy_model.py`, `misc.py`, `plot.py`, `lib.py`)

---

## 3. Dataset: dSprites

### 3.1 Download

Download the dSprites `.npz` file from DeepMind's GitHub:

```
URL: https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
Save to: data/toy/dsprites/dsprites.npz
```

### 3.2 Dataset Structure

The NPZ archive contains:
| Field | Shape | Description |
|---|---|---|
| `imgs` | (737280, 64, 64) | Binary images (uint8) |
| `latents_values` | (737280, 6) | Continuous latent factor values |
| `latents_classes` | (737280, 6) | Integer class indices for each factor |

### 3.3 Latent Factors

| Index | Factor | # Values | Range |
|---|---|---|---|
| 0 | Color | 1 | White (constant, ignore) |
| 1 | Shape | 3 | Square, Ellipse, Heart |
| 2 | Scale | 6 | [0.5, 1.0] |
| 3 | Rotation | 40 | [0, 2π] |
| 4 | Position X | 32 | [0, 1] |
| 5 | Position Y | 32 | [0, 1] |

### 3.4 Data Loading Function

Create a new function in `utils/data.py` (or a new file `data/toy/dsprites/dsprites_data.py`):

```python
def load_dsprites(path, n_samples=5000, seed=42):
    """
    Load dSprites dataset, subsample, flatten images, and split.
    
    Returns:
        X_train, X_test: flattened images (n, 4096), float32
        latents_train, latents_test: ground-truth latent values (n, 6)
        latents_classes_train, latents_classes_test: integer latent indices (n, 6)
    """
    data = np.load(path, allow_pickle=True)
    imgs = data['imgs']            # (737280, 64, 64)
    latents_values = data['latents_values']    # (737280, 6)
    latents_classes = data['latents_classes']   # (737280, 6)
    
    # Subsample (737K is too large for MLP + NTK analysis)
    np.random.seed(seed)
    indices = np.random.choice(len(imgs), size=n_samples, replace=False)
    
    imgs = imgs[indices]
    latents_values = latents_values[indices]
    latents_classes = latents_classes[indices]
    
    # Flatten images: (n, 64, 64) -> (n, 4096)
    X = torch.tensor(imgs.reshape(n_samples, -1), dtype=torch.float32)
    latents_values = torch.tensor(latents_values, dtype=torch.float32)
    latents_classes = torch.tensor(latents_classes, dtype=torch.long)
    
    # Train/test split (80/20)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    lv_train, lv_test = latents_values[:split], latents_values[split:]
    lc_train, lc_test = latents_classes[:split], latents_classes[split:]
    
    return X_train, X_test, lv_train, lv_test, lc_train, lc_test
```

> **Important Notes:**
> - Flatten images to 4096-dim vectors (required for MLP input)
> - Subsample to ~5000 samples (NTK Jacobian computation is O(n²) in memory; 737K is infeasible)
> - The `latents_values[:, 0]` (color) is constant — **exclude it** from analysis

---

## 4. Label Generation: Core & Spurious Mechanism

### 4.1 Design

We define a label-generating process with **known causal structure**:
- **Core features** → features that *causally determine* the label
- **Spurious features** → features that are *statistically correlated* with the label but not causal

### 4.2 Latent Factor Assignment

We use **2 core**, **2 spurious**, and **1 noise** latent factor:

| Role | Latent Factor | Index | Rationale |
|---|---|---|---|
| **Core 1** | Shape | 1 | Discrete (3 classes), defines object identity; harder to extract from raw pixels |
| **Core 2** | Scale | 2 | Continuous (6 values), defines object size; moderate difficulty |
| **Spurious 1** | Position X | 4 | Easy for MLP on flattened images (shifts which pixels are active) |
| **Spurious 2** | Position Y | 5 | Same as PosX; together they form a strong positional shortcut |
| **Noise** | Rotation | 3 | Varies freely, no correlation to label |

**Why this assignment?**
- Position (X, Y) is the easiest feature for an MLP to detect from flattened pixel data — a spatial shift directly changes which pixel indices are nonzero. This makes it a natural "shortcut" the model would prefer.
- Shape and Scale require learning higher-order structure (contours, area) that demands more network capacity, making them ideal "core" features.
- Rotation is independent of object identity and position, serving as genuine noise.

### 4.3 Label Generation Algorithm

```python
def generate_labels(latents_classes, latents_values,
                    core_indices=[1, 2],
                    spurious_indices=[4, 5],
                    spurious_correlation=0.9, seed=42):
    """
    Generate binary labels from 2 core features (Shape, Scale),
    with spurious correlation injected via 2 spurious features (PosX, PosY).
    
    Args:
        latents_classes: (n, 6) integer latent class indices
        latents_values: (n, 6) continuous latent factor values
        core_indices: indices of core latent factors [Shape=1, Scale=2]
        spurious_indices: indices of spurious latent factors [PosX=4, PosY=5]
        spurious_correlation: probability that spurious features agree with label
        seed: random seed
    
    Returns:
        y: (n,) binary labels {0, 1}
        core_labels: the "true" label from core features only (for analysis)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Step 1: Generate labels from BOTH core features jointly
    # Shape: 0=square, 1=ellipse, 2=heart
    # Scale: 6 values in [0.5, 1.0]
    # Strategy: combine Shape and Scale into a single score
    #   score = f(shape, scale) = shape_binary + scale_binary
    #   where shape_binary = 1 if {ellipse, heart}, 0 if square
    #   and scale_binary = 1 if scale > median, 0 otherwise
    #   label = 1 if score >= 1 (i.e., at least one core feature is "large"), else 0
    
    shape_classes = latents_classes[:, core_indices[0]]  # 0, 1, 2
    shape_binary = (shape_classes >= 1).long()            # square=0 vs {ellipse,heart}=1
    
    scale_values = latents_values[:, core_indices[1]]
    scale_median = scale_values.median()
    scale_binary = (scale_values > scale_median).long()   # small=0 vs large=1
    
    # Core label: OR of shape_binary and scale_binary
    # This means label=1 if the object is non-square OR large-scale (or both)
    # label=0 only if it is a small square
    core_score = shape_binary + scale_binary
    core_labels = (core_score >= 1).long()
    
    # Step 2: Inject spurious correlations via PosX and PosY
    # During TRAINING: with probability `spurious_correlation`,
    # ensure that label=1 samples tend to be in high PosX/PosY region
    # and label=0 samples tend to be in low PosX/PosY region.
    # Implementation: flip labels with probability (1 - spurious_correlation)
    y = core_labels.clone()
    n = len(y)
    flip_mask = torch.rand(n) > spurious_correlation
    y[flip_mask] = 1 - y[flip_mask]
    
    return y, core_labels
```

> **Note on the core label function**: The OR-based combination (`label=1 if non-square OR large-scale`) ensures both Shape and Scale are meaningful for the label. The model must learn to use *both* geometricproperties. Alternative functions (AND, XOR, weighted sum) can be explored as ablations.

### 4.4 Training vs. Evaluation Split Design

This is critical for testing OOD robustness:

- **Training set**: Labels have spurious correlation (p = 0.9 between PosX/PosY and label)
- **Evaluation set (ID)**: Same distribution as training (p = 0.9)
- **Evaluation set (OOD)**: Labels determined ONLY by core features Shape + Scale (p = 0.5 for spurious PosX/PosY)

> **Why this matters**: If the tunnel learns positional shortcuts (PosX, PosY), OOD accuracy will drop. If the extractor recovers the core features (Shape, Scale), CKA with core latents should peak at intermediate depth.

---

## 5. Model Training

### 5.1 Architecture

Follow the existing pattern from `train_latent.py`:

```python
input_dim = 4096  # flattened 64x64 images
hidden_dim = 120  # match existing experiments
activation_func = 'relu'

# Sweep over depths
depths = [1, 2, 3, 4, 5, 6, 7, 8]
```

Use the existing `Classifier` class from `utils/toy_model.py`:
```python
model = Classifier(input_dim, n_layer, hidden_dim, 
                   num_classes, activation_func)
```

Where `num_classes = 2` (binary classification).

### 5.2 Training Configuration

```python
train_dict = {
    'epochs': 1500,
    'lr': 0.02,
    'momentum': 0.9,
    'optimizer': 'sgd',
    'min_loss_change': 0.0001,
    'no_improve_threshold': 100,
    'loss_mp': 1,
    'wandb': False
}
```

### 5.3 Training Loop

For each depth in `depths`:
1. Initialize a new `Classifier` with that depth
2. Train using `train_model()` from `utils/misc.py`
3. Save the trained model (for reproducibility)
4. Verify that training loss converges (early stopping or epoch exhaustion)

### 5.4 Verification of Training

For each trained model, confirm:
- [ ] Training loss < 0.1 (or reaches early stopping criteria)
- [ ] Training accuracy > 90%
- [ ] Record both ID and OOD test accuracy

---

## 6. Post-Training Analysis

### 6.1 Layer-wise Latent CKA (Primary Analysis)

**Goal**: For each model depth, compute CKA between each hidden layer's output and the ground-truth latent factors.

**Procedure**:
1. Take the evaluation (test) set: `X_test`, `latents_test`
2. Feed `X_test` through the trained model
3. Extract each layer's output using `get_all_layer_outputs(model, X_test)` from `utils/misc.py`
4. For each layer output `H_l` (shape: `n × hidden_dim`):
   - Compute `CKA(H_l, latents_test[:, 1:])` — CKA with ALL latent factors (excluding color)
   - Compute `CKA(H_l, latents_test[:, [1, 2]])` — CKA with **CORE** factors (Shape + Scale)
   - Compute `CKA(H_l, latents_test[:, [4, 5]])` — CKA with **SPURIOUS** factors (PosX + PosY)
   - Compute `CKA(H_l, latents_test[:, 3:4])` — CKA with **NOISE** factor (Rotation)
5. Plot CKA vs. layer depth for each comparison (4 lines per plot)

**Use the existing `CudaCKA` class** from `utils/misc.py` (for GPU) or `CKA` class (for CPU).

> **Note**: The existing `layerwise_CKA()` function in `misc.py` already implements most of this logic. It can be adapted by changing the `latents` argument. With 2-column latent matrices (core or spurious), CKA is well-conditioned — no single-column degeneracy issues.

### 6.2 Additional Similarity Metrics

Beyond CKA, consider the following metrics for richer analysis:

#### a) Linear Probing Accuracy
- At each layer, train a small linear classifier (logistic regression) to predict each latent factor from the layer's output
- If latent `k` can be linearly predicted from layer `l` with high accuracy, it means layer `l` has "recovered" that latent
- This is more fine-grained than CKA because it measures **individual factor recovery**

```python
from sklearn.linear_model import LogisticRegression

def linear_probe(layer_output, latent_classes, factor_idx):
    """Train linear probe to predict a single latent factor."""
    X = layer_output.detach().cpu().numpy()
    y = latent_classes[:, factor_idx].cpu().numpy()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf.score(X, y)  # accuracy
```

#### b) Mutual Information (MI) Estimation
- Estimate MI between layer representations and latent factors
- Use a non-parametric estimator (e.g., k-NN based MI from `sklearn.feature_selection.mutual_info_classif`)
- MI is more general than CKA (captures nonlinear dependencies)

#### c) Representation Rank (already implemented)
- Use the existing `compute_layer_rank()` with `type='rep'` to track dimensionality
- Cross-reference rank peaks with CKA peaks

### 6.3 Expected Plots

Generate the following plots for each model depth (and as a combined summary):

| Plot | X-axis | Y-axis | Purpose |
|---|---|---|---|
| **A. CKA vs Layer (per model)** | Layer index | CKA score (4 lines: all, core, spurious, noise) | Shows where alignment with each latent group peaks |
| **B. CKA Peak Depth vs Model Depth** | Model depth (1-8) | Layer of peak CKA | Tests if extractor depth scales with model depth |
| **C. Core vs Spurious CKA** | Layer index | CKA score (2 lines: Shape+Scale vs PosX+PosY) | Shows if core/spurious are separated in representation space |
| **D. Linear Probe Accuracy** | Layer index | Accuracy (5 lines: Shape, Scale, Rotation, PosX, PosY) | Fine-grained per-factor recovery analysis |
| **E. Representation Rank** | Layer index | Soft rank | Cross-reference with CKA peaks |
| **F. ID vs OOD Accuracy** | Model depth | Test accuracy (2 lines) | Validates that depth hurts OOD performance |

---

## 7. Implementation Plan

### 7.1 New Files to Create

| File | Purpose |
|---|---|
| `data/toy/dsprites/dsprites_data.py` | Data loading, subsampling, label generation |
| `data/toy/dsprites/train_dsprites.py` | Main training & analysis script (follows `train_latent.py` pattern) |
| `data/toy/dsprites/__init__.py` | Empty init file |

### 7.2 Modifications to Existing Files

| File | Change |
|---|---|
| `utils/misc.py` | Add `latent_CKA_analysis()` function (adaptation of `layerwise_CKA` that computes CKA against individual latent factors and returns data instead of only plotting) |
| `utils/misc.py` | Add `linear_probe()` function for linear probing analysis |
| `utils/plot.py` | Add `plot_latent_cka_comparison()` for multi-line CKA plots |

### 7.3 Step-by-Step Execution Order

```
Step 1: Download dSprites dataset
        -> Save to data/toy/dsprites/dsprites.npz
        -> Verify file integrity (should be ~2.9 MB compressed)

Step 2: Create data/toy/dsprites/dsprites_data.py
        -> Implement load_dsprites()
        -> Implement generate_labels()
        -> Test: load data, print shapes, verify latent distributions

Step 3: Modify utils/misc.py
        -> Add latent_CKA_analysis() function
        -> Add linear_probe() function
        -> Test: run on a dummy model to verify output format

Step 4: Modify utils/plot.py
        -> Add plotting functions for the new analyses
        -> Test: generate dummy data and verify plots render

Step 5: Create data/toy/dsprites/train_dsprites.py
        -> Follow train_latent.py pattern
        -> Implement the depth sweep loop
        -> For each depth: train, analyze, save results

Step 6: Run the experiment
        -> Execute train_dsprites.py
        -> Monitor training convergence for each depth
        -> Save all generated plots to figs/dsprites/

Step 7: Verification
        -> Check training convergence for all depths
        -> Verify CKA plots are generated correctly
        -> Compare ID vs OOD accuracy across depths
        -> Document findings
```

---

## 8. Verification Checklist

### Training Verification
- [ ] dSprites data loads correctly (print sample image, verify shape = (n, 4096))
- [ ] Generated labels have correct core/spurious correlation (verify empirically: count agreement rate)
- [ ] Models train to convergence for all depths 1-8
- [ ] Training loss < 0.1 for at least depths 3+
- [ ] ID test accuracy > 85% for at least depths 3+

### Analysis Verification
- [ ] `get_all_layer_outputs()` returns the correct number of layers (== n_layer + 1)
- [ ] CKA values are in [0, 1] range
- [ ] CKA with all-latents shows a peak at intermediate depth (if hypothesis holds)
- [ ] Core CKA vs Spurious CKA show different depth profiles
- [ ] Linear probe accuracy for Shape and Scale (core) peaks at intermediate depth
- [ ] Linear probe accuracy for PosX and PosY (spurious) remains high across all layers
- [ ] Representation rank profile is consistent with previous experiments
- [ ] OOD accuracy is lower than ID accuracy for deeper models (if hypothesis holds)

### Sanity Checks
- [ ] CKA of input layer with latents should be low (images are nonlinear transform of latents)
- [ ] CKA of output layer with latents should be low (output is just 2-dim logits)
- [ ] For depth=1 model, there should be no "tunnel" behavior
- [ ] Random (untrained) model should show no CKA peak (control experiment)

---

## 9. Potential Issues & Mitigations

| Issue | Mitigation |
|---|---|
| **Memory**: NTK Jacobian on 4096-dim input is expensive | Reduce `n_samples` to 2000-3000; use `batch_size` in `compute_ntk()` |
| **MLP on images**: May not learn good features from flattened images | Use wider hidden layers (256-512); ensure training converges |
| **Spurious correlation too strong**: Model only learns PosX/PosY shortcut | Try `p = 0.8` or `p = 0.7` to give core features more chance |
| **Core label imbalance**: OR(shape, scale) may produce unbalanced labels | Verify class balance; if >70/30 split, switch to AND or adjust scale threshold |
| **Scale of latent values**: Different latent factors have different scales | Normalize each latent factor to zero mean, unit variance before CKA |

---

## 10. Dependencies

The experiment requires the following Python packages (most already present in `utils/lib.py`):
- `torch`, `numpy`, `matplotlib`, `seaborn` (already imported)
- `sklearn` (for linear probing — may need to install: `pip install scikit-learn`)
- `tqdm` (already imported)
- `functorch` (already imported)

---

## 11. Reference Files

These existing files should be studied before implementation:
- **Training pattern**: `data/toy/latent/train_latent.py` — the template to follow
- **CKA implementation**: `utils/misc.py` lines 111-265 — existing CKA classes and `layerwise_CKA()`
- **Layer output extraction**: `utils/misc.py` lines 453-474 — `get_all_layer_outputs()`
- **Rank analysis**: `utils/misc.py` lines 476-584 — `compute_layer_rank()`
- **Model definition**: `utils/toy_model.py` lines 57-103 — `Classifier` class
- **Data utilities**: `utils/data.py` — existing data generation patterns
- **Plotting**: `utils/plot.py` — existing plot styles and functions
