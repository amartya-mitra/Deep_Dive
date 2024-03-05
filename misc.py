from lib import *
from plot import *

# Ensure that the model and data tensor are on the same device
def ensure_same_device(model, data, device):
    """
    Ensure that the model and data tensor are on the same device.
    :param model: PyTorch model
    :param data: PyTorch tensor (input data)
    :return: model, data moved to the same device
    """
    # Check the device of the model
    model_device = next(model.parameters()).device
    
    # Check the device of the data tensor
    data_device = data.device
    
    # If model and data are on different devices, move them to the preferred device
    if model_device != data_device:
        model = model.to(device)
        data = data.to(device)
        
    return model, data

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

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def split_data(inputs):
  train_size = int(inputs.shape[0] * 0.6)
  test_size = int(inputs.shape[0] * 0.2)
  val_size = inputs.shape[0] - train_size - test_size
  return train_size, test_size, val_size

def train_model_loop(model, 
                     optimizer, 
                     criterion, 
                     train_dict, 
                     X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     X_val, 
                     y_val, 
                     epochs, 
                     use_early_stopping):
  
  # For plotting metrics
  train_losses, test_losses, val_losses = [], [], []
  train_errors, test_errors, val_errors = [], [], []

  # Calculate total steps for tqdm
  total_steps = epochs
  no_improve_threshold = train_dict['no_improve_threshold']
  min_loss_change = train_dict['min_loss_change']
  best_loss = float('inf')

  # Training loop with a single tqdm bar for all epochs
  with tqdm(total=total_steps, desc="Training Progress", position=0, leave=True) as pbar:
    for epoch in range(epochs):
      # Zero the parameter gradients
      model.train()
      optimizer.zero_grad()
      running_loss = 0.0

      # Forward pass
      outputs = model(X_train)
      loss = train_dict['loss_mp'] * criterion(outputs, (y_train + 1)/2)

      # Backward and optimize
      loss.backward()
      optimizer.step()

      ############## Compute Metrics ################
      train_outputs = model(X_train)
      train_loss = train_dict['loss_mp'] * criterion(train_outputs, (y_train + 1)/2)
      train_predictions = torch.sign(train_outputs)  # Convert probabilities to 0/1 predictions
      train_error = torch.mean((train_predictions != y_train).float())

      test_outputs = model(X_test)
      test_loss = train_dict['loss_mp'] * criterion(test_outputs, (y_test + 1)/2)
      test_predictions = torch.sign(test_outputs)  # Convert probabilities to 0/1 predictions
      test_error = torch.mean((test_predictions != y_test).float().float())

      val_outputs = model(X_val)
      val_loss = train_dict['loss_mp'] * criterion(val_outputs, (y_val + 1)/2)
      val_predictions = torch.sign(val_outputs)  # Convert probabilities to 0/1 predictions
      val_error = torch.mean((val_predictions != y_val).float().float())

      ################## Append Metrics ###############

      train_losses.append(train_loss.item())
      train_errors.append(train_error.item())

      test_losses.append(test_loss.item())
      test_errors.append(test_error.item())

      val_losses.append(val_loss.item())
      val_errors.append(val_error.item())

      if use_early_stopping:

        # Average loss for this epoch
        running_loss += loss.item()
        epoch_loss = running_loss

        # Check for improvement
        if best_loss - epoch_loss > min_loss_change:
            no_improve_epochs = 0
            best_loss = epoch_loss
        else:
            no_improve_epochs += 1

        # Early stopping check
        if no_improve_epochs >= no_improve_threshold:
            print(f'Early stopping triggered at epoch {epoch+1} with loss {epoch_loss:.4f}')
            break

      # # Print loss every 10 epochs
      # if (epoch+1) % 100 == 0:
      #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, 0-1 Error: {train_error.item():.4f}')

      # Update tqdm bar
      pbar.update(1)
      pbar.set_postfix_str(f'Loss: {loss.item():.4f}, 0-1 Error: {train_error.item():.4f}')

      # Log metrics to wandb
      wandb.log({"epoch": epoch, "loss": loss.item(), "accuracy": train_error.item()})

  return train_losses, test_losses, val_losses, train_errors, test_errors, val_errors

def train_model(model, epochs, use_early_stopping, use_gpu, train_dict, inputs, targets, seed=0):
  # Set seed for reproducibility
  set_seed(seed)

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

  # Early stopping parameters
  if use_early_stopping:
    early_stopping_params = {
        'min_loss_change': train_dict['min_loss_change'],  # Define 'substantial' change here
        'no_improve_threshold': train_dict['no_improve_threshold']  # Stop if no improvement in 5 epochs
    }

  # Loss function and optimizer
  criterion = nn.BCEWithLogitsLoss()
  if train_dict['optimizer'] == 'sgd':
     optimizer = optim.SGD(model.parameters(),
                        lr=train_dict['lr'],
                        momentum=train_dict['momentum'])

  elif train_dict['optimizer'] == 'adam':
     optimizer = optim.Adam(model.parameters(),
                        lr=train_dict['lr'])

  # Move inputs and targets to device
  inputs = inputs.to(device)
  targets = targets.to(device)

  # Split data into train, test, and validation sets
  train_size, test_size, val_size = split_data(inputs)
  X_train, y_train = inputs[:train_size].to(device), targets[:train_size].to(device)
  X_test, y_test = inputs[train_size:train_size+test_size].to(device), targets[train_size:train_size+test_size].to(device)
  X_val, y_val = inputs[train_size+test_size:].to(device), targets[train_size+test_size:].to(device)

  # Move model to device
  model.to(device)

  # Training loop
  train_losses, test_losses, val_losses, train_errors, test_errors, val_errors = train_model_loop(model, 
                                                                                                  optimizer, 
                                                                                                  criterion, 
                                                                                                  train_dict, 
                                                                                                  X_train, 
                                                                                                  y_train, 
                                                                                                  X_test, 
                                                                                                  y_test, 
                                                                                                  X_val, 
                                                                                                  y_val, 
                                                                                                  epochs, 
                                                                                                  use_early_stopping)
  
  # Plot and return trained model
  plot_metrics(train_losses, test_losses, val_losses, train_errors, test_errors, val_errors)
  return model

def mat_centering(mat):
  # 1. First, center the data by subtracting the mean
  mean_pred = torch.mean(mat, dim=0)
  centered_preds = mat - mean_pred
  return centered_preds

def get_entropy(mat):
  _, singular_values, _ = torch.svd(mat)
  tilde_sigma = singular_values / torch.sum(singular_values)
  shannon_entropy = - torch.dot(tilde_sigma, torch.log(tilde_sigma)).item()
  return shannon_entropy

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

def compute_layer_rank(model, activation_func, type, scale = False, input=None, use_gpu = False):

  if type == 'wgt':
    se = []
    le = []

    for name, module in model.named_modules():
        # Access only linear layers
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.data
            shannon_entropy = get_entropy(weights)
            se.append(shannon_entropy)
            le.append(int(name[6:]))
    plot_rank(se, le, xlabel='Depth', ylabel='Weight Rank', title=f'Wgt. Rank vs. Depth ({activation_func})')

  elif type == 'eff_wgt':
    composite_layer_matrix = []
    se = []
    le = []

    # Perform Singular Value Decomposition (SVD) and plot the singular value spectrum
    for name, module in model.named_modules():
        # Access only linear layers
        if isinstance(module, torch.nn.Linear):
          weights = module.weight.data
          if int(name[6:]) == 0:
            composite_layer_matrix.append(weights)
          else:
            composite_layer_matrix.append(torch.matmul(weights, composite_layer_matrix[-1]))

    for i in range(len(composite_layer_matrix)):
      shannon_entropy = get_entropy(composite_layer_matrix[i])
      se.append(shannon_entropy)
      le.append(i)
    plot_rank(se, le, xlabel='Depth', ylabel='Eff. Weight Rank', title=f'Eff. Wgt. Rank vs. Depth ({activation_func})')

  elif type == 'rep':
    device = 'cuda' if use_gpu else 'cpu'
    model, input = ensure_same_device(model, input, device)

    xlabel='Depth'
    ylabel='Representation Rank'
    title=f'Rep. Rank vs. Depth ({activation_func})'

    layer_ranks = []
    le = []
    counter = 0
    # Assuming 'model' is your trained model and 'input_data' is your dataset
    layer_outputs = get_all_layer_outputs(model, input)

    # Capture the output of each layer
    layer_modules = [input]
    for layer_name, layer_output in layer_outputs.items():
      layer_modules.append(layer_output)

    # Perform Singular Value Decomposition (SVD) of each layer output
    # and plot the singular value spectrum
    for layer_output in layer_modules:
      # 1. First, center the data by subtracting the mean
      centered_preds = mat_centering(layer_output)

      # 2. Then compute the covariance matrix
      cov_matrix = torch.matmul(centered_preds.T, centered_preds) / (centered_preds.shape[0] - 1)

      # 3. Then perform SVD of the above cov_matrix
      _, singular_values, _ = torch.svd(cov_matrix)

      # 4. Determine the maximum of the singular values (sigma_max)
      sigma_max = torch.max(singular_values)

      # 5. Set a threshold as 0.001 * sigma_max
      threshold = 0.001 * sigma_max

      # 6: Compute the number of singular values greater than or equal to the threshold
      rank = torch.sum(singular_values >= threshold).item()

      tilde_sigma = singular_values/torch.sum(singular_values)
      shannon_entropy = - torch.dot(tilde_sigma, torch.log(tilde_sigma))
      soft_rank = torch.exp(shannon_entropy).item()

      if scale:
        ylabel='Scaled Representation Rank'
        title=f'Scaled Rep. Rank vs. Depth ({activation_func})'

        if counter == 0:
          scale_factor = soft_rank
        soft_rank = soft_rank / scale_factor

      layer_ranks.append(soft_rank)
      le.append(counter)
      counter += 1

    plot_rank(layer_ranks, le, xlabel=xlabel, ylabel=ylabel, title=title)


  else:
    raise ValueError("Invalid type. Expected 'wgt'.")