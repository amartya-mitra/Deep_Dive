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

def generate_labels(latents_classes, latents_values,
                    core_indices=[1, 2],
                    spurious_indices=[4, 5],
                    spurious_correlation=0.9, seed=42):
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
    
    Returns:
        y: (n,) binary labels {0, 1}
        core_labels: the "true" label from core features only
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Step 1: Generate labels from BOTH core features jointly
    
    shape_classes = latents_classes[:, core_indices[0]]  # 0, 1, 2
    shape_binary = (shape_classes >= 1).long()            # square=0 vs {ellipse,heart}=1
    
    scale_values = latents_values[:, core_indices[1]]
    scale_median = scale_values.median()
    scale_binary = (scale_values > scale_median).long()   # small=0 vs large=1
    
    # Core label: OR of shape_binary and scale_binary
    core_score = shape_binary + scale_binary
    core_labels = (core_score >= 1).long()
    
    # Note: Spurious correlation logic is handled by dataset filtering in train_dsprites.py
    # We return core_labels as y.
    
    return core_labels, core_labels
