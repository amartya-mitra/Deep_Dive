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

######## Utility functions ########
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

def plot_ntk_md(results, x, n_layer_sizes):
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

######### Data generation and plotting #########
Args = namedtuple('Args', 'n_train lr max_iterations l2 loss flipped test_resolution dataset cmap')
cmap = 'plasma'
args = Args(dataset='disk', # ['disk', 'disk_flip_vertical', 'yinyang1', 'yinyang2']
            n_train=3000,
            lr=5e-2,
            max_iterations=9000,
            l2=0.,
            loss='ce',
            flipped=0.,
            test_resolution=60,
            cmap=cmap)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate dataset
r = (2 / np.pi)**.5
if args.dataset == 'yinyang1':
    x_train, y_train, x_test, y_test = generate_yinyang_dataset(n_train=args.n_train, p=args.flipped, test_resolution=args.test_resolution, variant=1)
elif args.dataset == 'yinyang2':
    x_train, y_train, x_test, y_test = generate_yinyang_dataset(n_train=args.n_train, p=args.flipped, test_resolution=args.test_resolution, variant=2)
elif args.dataset == 'disk':
    x_train, y_train, x_test, y_test = generate_disk_dataset(device, r, n_train=args.n_train, p=args.flipped, test_resolution=args.test_resolution)
elif args.dataset == 'disk_flip_vertical':
    x_train, y_train, x_test, y_test = generate_disk_dataset(device, r, n_train=args.n_train, p=args.flipped, test_resolution=args.test_resolution, half=True)

# Plot the dataset
plt.figure()

custom_imshow(plt.gca(), y_test, args)
plt.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
plt.xticks([]); plt.yticks([])
plt.show()

############### Training Config ###############
seed = 2
epochs = 5000
lr = 5e-4 # Normal:0.02, LH: 0.008
momentum = 0.1
min_loss_change = 0.0001
no_improve_threshold = 100
use_es = True
loss_mp = 1
activation_func = 'relu'
optimizer = 'adam'

train_dict = {'epochs': epochs,
              'min_loss_change': min_loss_change,
              'no_improve_threshold': no_improve_threshold,
              'use_es': use_es,
              'optimizer': optimizer,
              'lr': lr,
              'momentum': momentum,
              'loss_mp': loss_mp,
              'wandb': False}

hidden_dim = 50  # Hidden layer dimension
input_dim = x_train.shape[1]
features = x_train.to(torch.float32)
use_gpu = True if torch.cuda.is_available() else False
mode = 0

##############################
def train_model_depth(n_layer_sizes, epochs, use_es, use_gpu, train_dict, x_train, y_train, seed):
  results = []
  for n_layer in n_layer_sizes:
    model = Classifier(input_dim, n_layer, hidden_dim, len(torch.unique(y_train)), activation_func)

    if sum(p.numel() for p in model.parameters()) > len(x_train):
      print('Model of {} hidden layers is overparameterized'.format(n_layer))
    else:
      print('Model of {} hidden layers is underparameterized'.format(n_layer))

    model = train_model(model, epochs, use_es, use_gpu, train_dict, x_train.to(torch.float32), y_train, seed=0)
    results.append(model)
    print('Completed training of model with {} hidden layers'.format(n_layer))
  
  return results

n_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
# n_layer_sizes = [3]

results = train_model_depth(n_layer_sizes, epochs, use_es, use_gpu, train_dict, x_train, ((y_train + 1) / 2), seed)
# usvs = ntk_md(results, n_layer_sizes, input_dim, hidden_dim, x_train, y_train, x_test)
plot_ntk_md(results, x_test, n_layer_sizes)