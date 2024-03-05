import torch
import torch.nn as nn
import torch.optim as optim

from lib import *
from plot import *
from misc import *

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_width, activation_func):
        super(BinaryClassifier, self).__init__()

        # Create a list to hold the layers
        layers = []

        # Add the first hidden layer
        layers.append(nn.Linear(input_dim, hidden_layer_width))

        # Add the chosen activation function
        if activation_func == 'relu':
            layers.append(nn.ReLU())
        elif activation_func == 'tanh':
            layers.append(nn.Tanh())
        elif activation_func == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation_func == 'linear':
            layers.append(LinearActivation())
        elif activation_func == 'quadratic':
            layers.append(QuadraticActivation())
        else:
            raise ValueError("Unsupported activation function")

        # Add subsequent hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            if activation_func == 'relu':
                layers.append(nn.ReLU())
            elif activation_func == 'tanh':
                layers.append(nn.Tanh())
            elif activation_func == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_func == 'linear':
                layers.append(LinearActivation())
            elif activation_func == 'quadratic':
                layers.append(QuadraticActivation())

        # Add the output layer
        layers.append(nn.Linear(hidden_layer_width, 1))
        # layers.append(nn.Sigmoid())  # Assuming binary classification

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class NTK(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.fnet, self.params = make_functional(self.net)
        pc = torch.tensor([p.flatten().shape[0] for p in self.net.parameters()])
        pc = pc[::2] + pc[1::2]
        self.pc = torch.cumsum(pc, dim=0)

    def get_jac(self, x, device):
        # K: number of parameters blocks, e.g., 2 for Linear
        # n: number of examples in x
        # block_size: the shape of each param block
        # shape: K x n x out_dim x block_size
        jac = vmap(jacrev(self.fnet), (None, 0))(self.params, x.to(device))
        # shape: n x out_dim x num_all_params
        jac = torch.cat([j.flatten(2) for j in jac], 2)

        return jac.detach()

    def forward(self, jac):
        flat_params = torch.cat([p.flatten() for p in self.net.parameters()])
        return jac @ flat_params

    def to(self, device):
      self.net.to(device)  # Move the base model
      # Re-initialize functional parts that may depend on device-specific tensors
      self.fnet, self.params = make_functional(self.net)
      # Ensure tensors are reallocated on the correct device
      # This assumes self.params and any other custom tensors are not meta tensors
      # after re-initialization. If they are, additional handling will be needed.
      self.pc = self.pc.to(device)
      self.params = tuple(p.to(device) for p in self.params if p.is_cuda or p.device != device)
      return self