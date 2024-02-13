import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
# sys.path.append('.')
from lib import *
from data import *

# Number of samples
n_samples = 2000
feature_dict, seed, add_noise = init_config(n_samples)

X, y = get_toy_data(n_samples, feature_dict, seed, add_noise)