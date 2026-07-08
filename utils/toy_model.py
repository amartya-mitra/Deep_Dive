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

class Classifier(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_width, output_dim, activation_func, dropout=0.0):
        super(Classifier, self).__init__()

        # Create a list to hold the layers
        layers = []

        # Helper to get activation module
        def _get_activation():
            if activation_func == 'relu':
                return nn.ReLU()
            elif activation_func == 'tanh':
                return nn.Tanh()
            elif activation_func == 'sigmoid':
                return nn.Sigmoid()
            elif activation_func == 'linear':
                return LinearActivation()
            elif activation_func == 'quadratic':
                return QuadraticActivation()
            else:
                raise ValueError("Unsupported activation function")

        # Add the first hidden layer
        layers.append(nn.Linear(input_dim, hidden_layer_width))
        layers.append(_get_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Add subsequent hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(_get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Add the output layer
        layers.append(nn.Linear(hidden_layer_width, output_dim))

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        """
        Args:
            embed_dim : int — dimensionality of the output embedding vector.
        """
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x):
        """
        Args:
            x : FloatTensor (N, 4096) — flattened 64×64 grayscale images.
        Returns:
            embedding : FloatTensor (N, embed_dim)
        """
        x = x.view(x.size(0), 1, 64, 64)
        x = self.conv_blocks(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class CNNClassifier(nn.Module):
    def __init__(self, embed_dim, num_hidden_layers, hidden_layer_width,
                 output_dim, activation_func, dropout=0.0):
        """
        Args:
            embed_dim          : int — CNN encoder output dimension (input to MLP).
            num_hidden_layers  : int — number of hidden layers in the MLP.
            hidden_layer_width : int — width of each MLP hidden layer.
            output_dim         : int — number of output classes.
            activation_func    : str — passed through to Classifier.
            dropout            : float — passed through to Classifier.
        """
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.classifier = Classifier(
            input_dim=embed_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            output_dim=output_dim,
            activation_func=activation_func,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x : FloatTensor (N, 4096) — flat pixel input.
        Returns:
            logits : FloatTensor (N, output_dim)
        """
        embedding = self.encoder(x)
        return self.classifier(embedding)


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

    def get_layerwise_jac(self, x, device):
      # Layer-wise NTK
      lst_ntk = []
      jac = vmap(jacrev(self.fnet), (None, 0))(self.params, x.to(device))

      for id, j in enumerate(jac):
        loc = id // 2
        j = j.flatten(2)
        try:
          lst_ntk[loc] = torch.cat((lst_ntk[loc], j), 2)
        except IndexError:
          lst_ntk.append(j)

      return lst_ntk

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