import torch
import torch.nn as nn
import torch.optim as optim

from lib import *
from latent.latent import *

# Defining the binary classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim, n_layer, hidden_dim):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])  # First layer

        # Adding hidden layers
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output(x))
        return x

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
            pass
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

        # Add the output layer
        layers.append(nn.Linear(hidden_layer_width, 1))
        layers.append(nn.Sigmoid())  # Assuming binary classification

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)