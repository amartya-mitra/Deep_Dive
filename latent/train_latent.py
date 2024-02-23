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

# Number of samples
n_samples = 2000
feature_dict, seed, add_noise = init_config(n_samples)

# Generate data
X, y = get_toy_data(n_samples, feature_dict, seed, add_noise)

# Hyper-parameters
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

# Determine the regime of the model
if count_parameters(model) > X.shape[0]:
  print('The model is overparametrized')
else:
  print('The model is underparametrized')

# Training
model = train_model(model, epochs, use_es, use_gpu, train_dict, X, y.float().view(-1, 1), seed)

# Plot decision boundary
toy_plot(model, X, y, feature_dict, activation_func, seed)

# Observation:
# - For `linear` activations, even with a large number of additional noise dimensions, the decision boundary could be determined by the `core` features if it's strength/norm is comparable to the `spurious` features.
# - In other words, memorization in this toy setup is promoted by the presence of a significant number of noise dimensions AND weak in strength, `core` features
# - Note: Despite having stronger `spurious` features compared to the `core` features, the decision boundary is still determined by the `core` features. Upending the strength of the former, will eventually cause the decision boundary to be determined by the `spurious` features though

compute_layer_rank(model, activation_func, 'wgt')
compute_layer_rank(model, activation_func, 'eff_wgt')
compute_layer_rank(model, activation_func, 'rep', False, X)

# Compute CKA similarity
cka_similarity = layerwise_CKA(model, X, X, use_gpu)

# Observation:
# - The peak in the representation rank (with depth) occurs only if the model learns the `core` feature
# - The distance/depth of the peak in the representation rank (when it occurs) from the input layer, depends on the learnability of the `core` feature. I.e. if the `core` feature is not well learned, then the peak in the representation rank will be close to the input layer, and vice versa.
# - **This raises a new question. How does the hypothesis of class sample diversity leading to deeper peaking of the representation rank, connect to the above?**
#  - It kind of makes sense. The smaller the *tunnel*, the less the `spurious` features are learned. Now, diversity makes the model less prone to learn the `spurious` features. And hence, the supposed observation is that the representation rank will be deeper in the model.