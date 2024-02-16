from lib import *

# Activation functions
class LinearActivation(nn.Module):
    def forward(self, x):
        return x  # Identity function

class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x**2  # Quadratic function

class jac_NTK():
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

def compute_ntk(x, model, use_gpu, batch_size=100):
    # Compute the Jacobian matrix
    jac_ntk = jac_NTK(model)
    jac = jac_ntk.get_jac(x)

    # Compute the NTK matrix
    

    """Compute the dot product in batches to reduce memory usage."""
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    n = jac.shape[0]
    jac.to(device)
    result = torch.zeros((n, n), device=jac.device)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            result[i:end_i, j:end_j] = torch.matmul(jac[i:end_i].squeeze(1), jac[j:end_j].squeeze(1).transpose(0, 1))

    return result

def quadratic_activation(tensor):
    """Applies a quadratic activation function element-wise."""
    return tensor ** 2

# Code to determine number of trainable parameters in the model
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)