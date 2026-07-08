import numpy as np
import torch

def load_dsprites(path, n_samples=5000, seed=42):
    """
    Load dSprites dataset, subsample, flatten images, and split.
    
    Returns:
        X_train, X_test: flattened images (n, 4096), float32
        latents_train, latents_test: ground-truth latent values (n, 6)
        latents_classes_train, latents_classes_test: integer latent indices (n, 6)
    """
    try:
        data = np.load(path, allow_pickle=True, encoding='latin1')
    except Exception as e:
        print(f"Error loading dSprites from {path}: {e}")
        raise

    imgs = data['imgs']            # (737280, 64, 64)
    latents_values = data['latents_values']    # (737280, 6)
    latents_classes = data['latents_classes']   # (737280, 6)
    
    # Subsample (737K is too large for MLP + NTK analysis)
    np.random.seed(seed)
    if n_samples > len(imgs):
        print(f"Warning: n_samples ({n_samples}) > dataset size ({len(imgs)}). Using full dataset.")
        indices = np.arange(len(imgs))
    else:
        indices = np.random.choice(len(imgs), size=n_samples, replace=False)
    
    imgs = imgs[indices]
    latents_values = latents_values[indices]
    latents_classes = latents_classes[indices]
    
    # Flatten images: (n, 64, 64) -> (n, 4096)
    X = torch.tensor(imgs.reshape(len(indices), -1), dtype=torch.float32)
    latents_values = torch.tensor(latents_values, dtype=torch.float32)
    latents_classes = torch.tensor(latents_classes, dtype=torch.long)
    
    # Train/test split (80/20)
    split = int(0.8 * len(indices))
    X_train, X_test = X[:split], X[split:]
    lv_train, lv_test = latents_values[:split], latents_values[split:]
    lc_train, lc_test = latents_classes[:split], latents_classes[split:]
    
    return X_train, X_test, lv_train, lv_test, lc_train, lc_test

def generate_labels_6class(latents_classes, latents_values):
    """
    Generate 6-class labels from Shape (3 values) x Scale bin (2 values).

    Class layout:
        class_id = shape * 2 + scale_bin
        0: Square  + Small    3: Ellipse + Large
        1: Square  + Large    4: Heart   + Small
        2: Ellipse + Small    5: Heart   + Large

    Args:
        latents_classes : LongTensor (N, 6) — integer latent class indices
        latents_values  : FloatTensor (N, 6) — continuous latent values

    Returns:
        labels    : LongTensor (N,), values in {0,1,2,3,4,5}
        scale_bin : LongTensor (N,), values in {0,1}  (0=small, 1=large)
    """
    scale_values = latents_values[:, 2]
    scale_bin = (scale_values > scale_values.median()).long()
    shape = latents_classes[:, 1]          # 0=Square, 1=Ellipse, 2=Heart
    labels = shape * 2 + scale_bin
    return labels, scale_bin


def filter_dataset_for_bias_6class(
    X, labels, scale_bin, latents_classes, latents_values,
    p_correlation, use_spurious, seed=42
):
    """
    Filter a 6-class dSprites split to inject spurious correlation.

    The spurious feature (PosX AND PosY > median) is aligned with scale_bin.
    Misaligned examples are undersampled until P(aligned) = p_correlation.
    No class-balance correction is applied (6-class labels are balanced by
    construction under uniform dSprites sampling).

    Args:
        X               : FloatTensor (N, input_dim)
        labels          : LongTensor  (N,) — 6-class labels from generate_labels_6class
        scale_bin       : LongTensor  (N,) — binary scale component (alignment target)
        latents_classes : LongTensor  (N, 6)
        latents_values  : FloatTensor (N, 6)
        p_correlation   : float in (0.5, 1.0] — desired P(spurious aligned with scale_bin)
        use_spurious    : bool — master toggle; if False, return inputs unchanged
        seed            : int

    Returns:
        X_filtered, labels_filtered, latents_classes_filtered, latents_values_filtered
    """
    if not use_spurious:
        return X, labels, latents_classes, latents_values

    np.random.seed(seed)

    pos_x = latents_values[:, 4]
    pos_y = latents_values[:, 5]
    spurious_bit = ((pos_x > pos_x.median()).long() +
                    (pos_y > pos_y.median()).long() == 2).long()

    aligned = (scale_bin == spurious_bit).cpu().numpy()
    indices = np.arange(len(labels))
    aligned_indices    = indices[aligned]
    misaligned_indices = indices[~aligned]

    n_aligned    = len(aligned_indices)
    n_misaligned = len(misaligned_indices)

    n_keep_mis = int(n_aligned * (1 - p_correlation) / p_correlation)
    if n_keep_mis < n_misaligned:
        keep_mis = np.random.choice(misaligned_indices, n_keep_mis, replace=False)
        final_indices = np.concatenate([aligned_indices, keep_mis])
    else:
        final_indices = np.concatenate([aligned_indices, misaligned_indices])

    np.random.shuffle(final_indices)
    return (X[final_indices], labels[final_indices],
            latents_classes[final_indices], latents_values[final_indices])


def generate_labels(latents_classes, latents_values,
                    core_indices=[1, 2],
                    spurious_indices=[4, 5],
                    spurious_correlation=0.9, seed=42, logic='OR'):
    """
    Generate binary labels from 2 core features (Shape, Scale).
    Returns y (core labels) and core_labels (identical).
    
    Args:
        latents_classes: (n, 6) integer latent class indices
        latents_values: (n, 6) continuous latent factor values
        core_indices: indices of core latent factors [Shape=1, Scale=2]
        spurious_indices: indices of spurious latent factors [PosX=4, PosY=5]
        spurious_correlation: Not used directly here, handled by dataset filtering.
        seed: random seed
        logic: 'OR' or 'AND'. Defines how core features are combined.
    
    Returns:
        y: (n,) binary labels {0, 1}
        core_labels: the "true" label from core features only
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Step 1: Generate labels from BOTH core features jointly
    
    shape_classes = latents_classes[:, core_indices[0]]  # 0, 1, 2
    
    # Shape Binary: 
    # If logic is OR (original), 1=Ellipse/Heart.
    # If logic is AND (filtered), we dropped Ellipse(1), so we care about Heart(2).
    # Being explicit: Square=0.
    if logic == 'AND':
        # Assuming Ellipses are dropped, so we distinguish Heart(2) vs Square(0).
        # We treat only Heart as positive.
        shape_binary = (shape_classes == 2).long()
    else:
        # Original: Ellipse(1) + Heart(2) are positive.
        shape_binary = (shape_classes >= 1).long()
    
    scale_values = latents_values[:, core_indices[1]]
    scale_median = scale_values.median()
    scale_binary = (scale_values > scale_median).long()   # small=0 vs large=1
    
    # Core label
    if logic == 'AND':
         core_score = shape_binary + scale_binary
         core_labels = (core_score == 2).long() # Both must be 1
    else:
        # OR
        core_score = shape_binary + scale_binary
        core_labels = (core_score >= 1).long()
    
    # Note: Spurious correlation logic is handled by dataset filtering in train_dsprites.py
    # We return core_labels as y.
    
    return core_labels, core_labels
