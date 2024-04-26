import sys
import os

# Path to the directory containing the utility files
utility_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils', ''))
sys.path.append(utility_directory)
from lib import *
from data import *
from toy_model import *
from plot import *
from misc import *
from collections import namedtuple
from re import U

def ntk_md(results, n_layer_sizes, input_dim, hidden_dim, x_train, y_train, x_test = None):
   usvs = []
   for idx, n_layer in enumerate(n_layer_sizes):
    print(f'Computing eigendecomposition of the NTRF matrix of a {n_layer}-layer model')

    ##############################
    reset = False # Reset the model to initial state

    if reset == True:
      reset_model = Classifier(input_dim, n_layer, hidden_dim, len(torch.unique(y_train)), activation_func)
    else:
      reset_model = results[idx]
      reset_model.load_state_dict(results[idx].state_dict())

    ntk = NTK(reset_model)

    if x_test is not None:
       inputs = ntk.get_jac(x_test, next(ntk.parameters()).device)
    else:
       inputs = ntk.get_jac(x_train, next(ntk.parameters()).device)
    ##############################
    u, s, v = torch.svd(inputs[:, 0, :].t())
    ##############################

    print(f'Shape of U, S, V: {u.shape[0], u.shape[1]}, ({s.shape[0]}), {v.shape[0], v.shape[1]}')
    usvs.append((u, s, v))
    
    return usvs

def get_rank(A, hard=False):
    if hard:
        return torch.linalg.matrix_rank(A.detach())
    else:
        U, S, V = torch.svd(A.detach())
        # Normalize the singular values
        S = S / S.sum()
        entropy = -torch.sum(S * torch.log(S))
        soft_rank = torch.exp(entropy).item()
        # soft_rank = entropy.item()
        return soft_rank

inputs = [] # Contains the Jacobian matrices of the separate models
layerwise_ntk = [] # Contains the layerwise Jacobian matrices of the separate models
NTRF_rank = []

for idx, n_layer in enumerate(n_layer_sizes):
  ntk = NTK(results[idx])
  inputs.append(ntk.get_jac(x_test, next(ntk.parameters()).device))
  layerwise_ntk.append(ntk.get_layerwise_jac(x_test, next(ntk.parameters()).device))

for ele in inputs:
  NTRF = torch.matmul(ele[:, 0, :], ele[:, 0, :].t())
  NTRF_rank.append(get_rank(NTRF, hard=False))

plt.plot(n_layer_sizes, NTRF_rank, marker='o', color='b')
plt.xlabel('# of hidden layers')
plt.ylabel('Rank of the NTK matrix')
plt.show()