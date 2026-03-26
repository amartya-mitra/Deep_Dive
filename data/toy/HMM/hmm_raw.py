# Raw features generated as (Latent --> Add Noise dimensions)

import sys
import os
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
# sys.path.append('.')
from lib import *
from data import *
from toy_model import *
from plot import *
from misc import *

# Setup the data distribution
data_config = {
    "nonlin": torch.sigmoid,
    "dim_latent": 4,
    "dim_ambient": 512,
    "n": 2000,
    "p": 0.5,
    "noise": 0.0
    }

data = get_HMM_data(data_config)

# Generate data
X = data['tr']['x']
y = data['tr']['y']

X = X.to(torch.float32)
y = y.to(torch.float32)

y = y.view(-1, 1)

###############
# Defining hyper-parameters
epochs = 1500
lr = 0.01 # Normal:0.02, LH: 0.008
momentum = 0.9
min_loss_change = 0.0001
no_improve_threshold = 100
use_es = True
loss_mp = 1
activation_func = 'relu'
optimizer = 'sgd'

train_dict = {'epochs': epochs,
              'min_loss_change': min_loss_change,
              'no_improve_threshold': no_improve_threshold,
              'use_es': use_es,
              'optimizer': optimizer,
              'lr': lr,
              'momentum': momentum,
              'loss_mp': loss_mp}

n_layer = 5  # Number of layers (Normal: 5, LH: 1) 
hidden_dim = 120 # Hidden layer dimension
input_dim = X.shape[1]
X = X.to(torch.float32)
use_gpu = torch.cuda.is_available()
mode = 0

##### Wandb Config #####

config = {**data_config, 
          **train_dict,
          'n_layer': n_layer,
          'hidden_dim': hidden_dim,
          'input_dim': input_dim,
          'activation_func': activation_func,
          'optimizer': optimizer,
          'use_es': use_es,
          'use_gpu': use_gpu,
          'mode': mode
          }


wandb.init(project='DeepDive', entity='amartya-mitra', config=config)

model = BinaryClassifier(input_dim, n_layer, hidden_dim, activation_func)

# standard rich training
if mode == 0:
    model = train_model(model, 
                        epochs, 
                        use_es, 
                        use_gpu, 
                        train_dict, 
                        X, y.float().view(-1, 1), 
                        seed = 2)
    
    # Plot the layer ranks
    compute_layer_rank(model, activation_func, 'wgt')
    compute_layer_rank(model, activation_func, 'eff_wgt')
    compute_layer_rank(model, activation_func, 'rep', False, X)
    
    # Extract the latent features
    latent_X = data['tr']['latents']
    
    # Compute CKA similarity
    cka_similarity = layerwise_CKA(model, X, latent_X, use_gpu)
    
# Lazy training
elif mode == 1:
    ntk = NTK(model)
    inputs = ntk.get_jac(X, next(ntk.parameters()).device)

    ntk = train_model(ntk,
                      epochs,
                      use_es, 
                      use_gpu,
                      train_dict, 
                      inputs,
                      y.float().view(-1, 1),
                      seed)
    
    
    # Plot the layer ranks
    compute_layer_rank(model, activation_func, 'wgt')
    compute_layer_rank(model, activation_func, 'eff_wgt')
    compute_layer_rank(model, activation_func, 'rep', False, X)