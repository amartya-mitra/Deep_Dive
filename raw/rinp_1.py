import sys
import os
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
# sys.path.append('.')
from lib import *
from data import *
from toy_model import *
from plot_rank import *

# Setup the data distribution
# add noise?
add_noise = True
# Noise dampener?
noise_dampener = 1 # 10: Memorize, 1: Generalize
# Noise strength
noise_multiplier = 0.1

n_samples = 2000
seed = 2

feature_dict = {'mu': mu,
                'sigma': sigma,
                'p': p,
                'spurious_multiplier': spurious_multiplier,
                'core_multiplier': core_multiplier,
                'noise_multiplier': noise_multiplier,
                'noise_dampener': noise_dampener}

X, y = get_toy_data(n_samples, feature_dict, seed, add_noise)