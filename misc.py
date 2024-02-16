from lib import *

# Activation functions
class LinearActivation(nn.Module):
    def forward(self, x):
        return x  # Identity function

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x**2  # Quadratic function

class NTK():
    def __init__(self, net):
        self.fnet, self.params = make_functional(net)

    def get_jac(self, x):
        # K: number of parameters blocks, e.g., 2 for Linear
        # n: number of examples in x
        # block_size: the shape of each param block
        # shape: K x n x out_dim x block_size
        jac = vmap(jacrev(self.fnet), (None, 0))(self.params, x)
        # shape: n x out_dim x num_all_params
        jac = torch.cat([j.flatten(2) for j in jac], 2)

        return jac
    
def quadratic_activation(tensor):
    """Applies a quadratic activation function element-wise."""
    return tensor ** 2

# Code to determine number of trainable parameters in the model
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)