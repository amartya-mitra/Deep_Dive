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

def get_all_layer_outputs(model, input_data):
    # Dictionary to store layer outputs
    layer_outputs = {}

    # Function to be called by the hook
    def hook_fn(module, input, output):
        layer_outputs[module] = output

    # Register hooks on all linear layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn))

    # Perform a forward pass with the input data to trigger the hooks
    model(input_data)

    # Remove hooks after use to avoid memory leaks
    for hook in hooks:
        hook.remove()

    return layer_outputs

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)
  
class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

def layerwise_CKA(model, input, latents, use_gpu):
    plt.style.use("seaborn-v0_8-pastel")  # Or choose a different style you prefer
    plt.figure(figsize=(5, 5))
    
    layer_CKA = []
    le = []
    counter = 0
    # Assuming 'model' is your trained model and 'input_data' is your dataset
    layer_outputs = get_all_layer_outputs(model, input)
    device = torch.device('cuda' if use_gpu else torch.device('cpu'))

    if use_gpu == False:
        cka = CKA()
    else:
        cka = CudaCKA(device)

    # Create 2D Torch Tensor of dimension len(layer_outputs)
    cka_corr = torch.zeros(len(layer_outputs), len(layer_outputs))

    ###########################################################################
#   Compute the individual CKA values of each layer w.r.t. the latent
    for layer_name, layer_output in layer_outputs.items():
        # Perform CKA
        if counter == 0:
            # print(f'Linear CKA at layer {counter}:', cka.linear_CKA(input.to(device), latents.to(device)).item())
            # print(f'RBF Kernel CKA at layer {counter}:', cka.kernel_CKA(input.to(device), latents.to(device)).item())
            layer_CKA.append(cka.kernel_CKA(input.to(device), latents.to(device)).item())
            counter += 1
        
    #     # print(f'Linear CKA at layer {counter}:', cka.linear_CKA(layer_output.to(device), latents.to(device)).item())
    #     # print(f'RBF Kernel CKA at layer {counter}:', cka.kernel_CKA(layer_output.to(device), latents.to(device)).item())
        layer_CKA.append(cka.kernel_CKA(layer_output.to(device), latents.to(device)).item())
        counter += 1
    
    ys = layer_CKA
    xs = [x for x in range(len(ys))]

    plt.plot(xs, ys, label="Latent-layer Representation Similarity", linestyle='solid', color='red')
    # Customize axes, title, and background
    plt.xlabel('Layer #', fontweight='light')
    plt.ylabel('Latent-layer Representation Similarity', fontweight='light')
    plt.grid(True)  # Turn on grid
    plt.gca().set_facecolor('lightgray')  # Set plot background color
    
    plt.legend()
    plt.show()
    ###########################################################################
#   Compute the inter-layer CKA values of the network
    i = 0
    for layer_name_i, layer_output_i in layer_outputs.items():
        j = 0
        for layer_name_j, layer_output_j in layer_outputs.items():
            cka_corr[i][j] = cka.kernel_CKA(layer_output_i.to(device), layer_output_j.to(device))
            j += 1
        i += 1
    
    # Plot the heatmap
    fig = sns.heatmap(cka_corr.detach().cpu().numpy(), 
                    #  xticklabels=layer_outputs.keys(), 
                    #  yticklabels=layer_outputs.keys(), 
                     linewidth=0.5, 
                     cmap = sns.cm.rocket_r)
    fig.set_title('Inter-layer Representation Similarity')
    plt.show()
    ###########################################################################