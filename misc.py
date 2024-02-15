from lib import *

def quadratic_activation(tensor):
    """Applies a quadratic activation function element-wise."""
    return tensor ** 2

# Code to determine number of trainable parameters in the model
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)