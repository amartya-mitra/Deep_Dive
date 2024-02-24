# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

from lib import *
from misc import *

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

    '''
#   Compute the individual CKA values of each layer w.r.t. the latent
    for layer_name, layer_output in layer_outputs.items():
        # Perform CKA
        if counter == 0:
            # print(f'Linear CKA at layer {counter}:', cka.linear_CKA(input.to(device), latents.to(device)).item())
            print(f'RBF Kernel CKA at layer {counter}:', cka.kernel_CKA(input.to(device), latents.to(device)).item())
            print()
            counter += 1
        
        # print(f'Linear CKA at layer {counter}:', cka.linear_CKA(layer_output.to(device), latents.to(device)).item())
        print(f'RBF Kernel CKA at layer {counter}:', cka.kernel_CKA(layer_output.to(device), latents.to(device)).item())
        print()
        counter += 1
#   layer_modules.append(layer_output)
    '''
    i = 0
#   Compute the inter-layer CKA values of the network
    for layer_name_i, layer_output_i in layer_outputs.items():
        j = 0
        for layer_name_j, layer_output_j in layer_outputs.items():
            cka_corr[i][j] = cka.kernel_CKA(layer_output_i.to(device), layer_output_j.to(device))
            j += 1
        i += 1
    
    # Plot the heatmap
    ax = sns.heatmap(cka_corr.detach().cpu().numpy(), linewidth=0.5, cmap = sns.cm.rocket_r)
    plt.show()