
import numpy as np
import torch
import os

def inspect_dsprites():
    path = 'data/toy/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    try:
        data = np.load(path, allow_pickle=True, encoding='latin1')
    except Exception as e:
        print(f"Error: {e}")
        return

    latents_values = data['latents_values'] # (N, 6)
    latents_names = ['Color', 'Shape', 'Scale', 'Orientation', 'PosX', 'PosY']

    print(f"Dataset Size: {len(latents_values)}")
    for i, name in enumerate(latents_names):
        unique_vals = np.unique(latents_values[:, i])
        print(f"\nFeature {i}: {name}")
        print(f"  Count: {len(unique_vals)}")
        if len(unique_vals) < 20:
            print(f"  Values: {unique_vals}")
        else:
             print(f"  Values (first 5): {unique_vals[:5]} ... (last 5): {unique_vals[-5:]}")

if __name__ == "__main__":
    inspect_dsprites()
