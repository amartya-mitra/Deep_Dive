# Raw features generated as (Latent --> Add Noise dimensions)

import sys
import os
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
# sys.path.append('.')
from lib import *
from data import *
from toy_model import *
from plot_rank import *
from misc import *
from CKA import *

# Setup the data distribution
# add noise?
add_noise = True
# Noise dampener?
noise_dampener = 1 # 10: Memorize, 1: Generalize
# Noise strength
noise_multiplier = 0.1

# Number of samples
n_samples = 2000
feature_dict, seed, add_noise = init_config(n_samples)
# Reset the add_noise parameter
add_noise = True

# Generate data
X, y = get_toy_data(n_samples, feature_dict, seed, add_noise)

###############
# Defining hyper-parameters
epochs = 1500
lr = 0.02 # Normal:0.02, LH: 0.008
momentum = 0.9
min_loss_change = 0.0001
no_improve_threshold = 100
use_es = True
loss_mp = 1
activation_func = 'linear'

train_dict = {'epochs': epochs,
              'min_loss_change': min_loss_change,
              'no_improve_threshold': no_improve_threshold,
              'use_es': use_es,
              'lr': lr,
              'momentum': momentum,
              'loss_mp': loss_mp}

n_layer = 5  # Number of layers (Normal: 5, LH: 1)
hidden_dim = 120  # Hidden layer dimension
input_dim = X.shape[1]
X = X.to(torch.float32)
use_gpu = True

# Creating the model instance
model = BinaryClassifier(input_dim, n_layer, hidden_dim, activation_func)
model = train_model(model, epochs, use_es, use_gpu, train_dict, X, y.float().view(-1, 1), seed)

# Plot the decision boundary
toy_plot(model, X, y, feature_dict, activation_func, seed)

# Plot the layer ranks
compute_layer_rank(model, activation_func, 'wgt')
compute_layer_rank(model, activation_func, 'eff_wgt')
compute_layer_rank(model, activation_func, 'rep', False, X)

cka_similarity = layerwise_CKA(model, X, use_gpu)