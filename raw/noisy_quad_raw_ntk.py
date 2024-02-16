# Raw features generated as (Latent --> Add Noise dimensions --> Rotated core + spurious features)

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
activation_func = 'relu'
use_gpu = True
print('Activation function: ', activation_func)

train_dict = {'epochs': epochs,
              'min_loss_change': min_loss_change,
              'no_improve_threshold': no_improve_threshold,
              'use_es': use_es,
              'lr': lr,
              'momentum': momentum,
              'loss_mp': loss_mp}

n_layer = 5  # Number of layers (Normal: 5, LH: 1)
hidden_dim = 120  # Hidden layer dimension
X = X.to(torch.float32)

# Generate a 2x2 random tensor
random_tensor = torch.randn(2, 2)
# random_tensor = torch.tensor([[1, 0], [0, 1]], dtype=torch.float) # Unit tensor
random_tensor = torch.tensor(ortho_group.rvs(dim=2), dtype=torch.float)

# Multiply the input tensor with the 2x2 random tensor
# This operation conforms to matrix multiplication rules, resulting in a n x 2 tensor
X_m = torch.mm(X[:, :2], random_tensor)
# X_m = torch.mm(X_m, random_tensor.T)

# Apply a quadratic activation on top
X_m = quadratic_activation(X_m)
X_m = torch.cat((X_m, X[:, 2:]), dim=1)

#############################################
n_layer = 5  # Number of layers (Normal: 5, LH: 1)
hidden_dim = 120  # Hidden layer dimension
input_dim = X_m.shape[1]
use_gpu = True

# Creating the model instance
model = BinaryClassifier(input_dim, n_layer, hidden_dim, activation_func)
jac_ntk = jac_NTK(model)
jac = jac_ntk.get_jac(X_m)
print(jac.shape)

# Computing the NTK matrix
ntk = batched_NTK(jac, use_gpu)
print(ntk.shape)

# model = train_model(model, epochs, use_es, use_gpu, train_dict, X_m, y.float().view(-1, 1))

# # Plot the decision boundary
# toy_plot(model, X_m, y, feature_dict, activation_func, seed)

# # # Plot the layer ranks
# compute_layer_rank(model, activation_func, 'wgt')
# compute_layer_rank(model, activation_func, 'eff_wgt')
# compute_layer_rank(model, activation_func, 'rep', False, X_m)