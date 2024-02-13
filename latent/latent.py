# Import packages from lib.py
import sys
sys.path.append('..')
import toy_data

# Number of samples
n_samples = 2000
feature_dict, seed, add_noise = init_config(n_samples)

X, y = get_toy_data(n_samples, feature_dict, seed, add_noise)