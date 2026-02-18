"""
Yin-Yang Dataset Generator

Generates a 2D binary classification dataset based on the Yin-Yang pattern:
- Top half (y > 0): Class 0, except a circle -> Class 1
- Bottom half (y <= 0): Class 1, except a circle -> Class 0

This creates a non-linearly separable problem.
"""

import numpy as np
import torch


def generate_yinyang(n_samples=10000, seed=42):
    """
    Generate the Yin-Yang dataset.
    
    Args:
        n_samples: Number of samples to generate.
        seed: Random seed.
    
    Returns:
        X: (n_samples, 2) tensor of (x, y) coordinates in [-1, 1]^2.
        y: (n_samples,) tensor of binary labels {0, 1}.
    """
    np.random.seed(seed)
    
    # Sample uniformly in [-1, 1]^2
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    
    # Define the Yin-Yang rule
    # Based on the reference image:
    # - Top half (y > 0): predominantly Class 0 (blue / dark)
    #   - Circle centered at (-0.35, 0.5), radius 0.3: Class 1 (yellow / light)
    # - Bottom half (y <= 0): predominantly Class 1 (yellow / light)
    #   - Circle centered at (0.35, -0.5), radius 0.3: Class 0 (blue / dark)
    
    circle1_center = np.array([-0.35, 0.5])
    circle1_radius = 0.30
    circle2_center = np.array([0.35, -0.5])
    circle2_radius = 0.30
    
    labels = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        x, y_coord = X[i]
        
        # Check circle membership
        in_circle1 = np.sum((X[i] - circle1_center) ** 2) < circle1_radius ** 2
        in_circle2 = np.sum((X[i] - circle2_center) ** 2) < circle2_radius ** 2
        
        if y_coord > 0:
            # Top half: Class 0, unless in circle1 -> Class 1
            labels[i] = 1 if in_circle1 else 0
        else:
            # Bottom half: Class 1, unless in circle2 -> Class 0
            labels[i] = 0 if in_circle2 else 1
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    return X_tensor, y_tensor
