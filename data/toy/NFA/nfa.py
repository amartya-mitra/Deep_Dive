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

# # Function to print the contents of a directory
# def print_directory_contents(directory):
#     try:
#         # Get the list of files and directories in the specified directory
#         contents = os.listdir(directory)
        
#         # Print each file and directory
#         for item in contents:
#             print(item)
#     except FileNotFoundError:
#         print("Directory not found.")

# # Specify the directory path
# directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils', ''))

# # Call the function to print the directory contents
# print_directory_contents(directory_path)

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
epochs = 9000
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

n_layer = 5  # Number of hidden layers (Normal: 5, LH: 1)
hidden_dim = 50  # Hidden layer dimension
input_dim = x_train.shape[1]
features = x_train.to(torch.float32)
use_gpu = True if torch.cuda.is_available() else False
mode = 0

model = Classifier(input_dim, n_layer, hidden_dim, len(torch.unique(y_train)), activation_func)

if sum(p.numel() for p in model.parameters()) > len(x_train):
    print('Model of {} hidden layers is overparameterized'.format(n_layer))
else:
    print('Model of {} hidden layers is underparameterized'.format(n_layer))

model = train_model(model,
                    epochs,
                    use_es,
                    use_gpu,
                    train_dict,
                    features,
                    ((y_train + 1)/2),
                    # y.float().view(-1, 1),
                    seed)

###### Temp plot ######
if len(torch.unique(y_train)) <= 2:
    toy_plot(model, x_train, y_train, activation_func, seed)

