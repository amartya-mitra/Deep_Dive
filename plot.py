from lib import *

def plot_metrics(train_losses, test_losses, val_losses, train_errors, test_errors, val_errors):
  # Set modern style for plot elements
  plt.style.use("seaborn-v0_8-pastel")  # Or choose a different style you prefer

  plt.figure(figsize=(12, 6))  # Adjust figure size as needed

  # Loss Curves
  ax1 = plt.subplot(121)
  ax1.plot(train_losses, label='Training Loss', linestyle='--', color='blue')
  ax1.plot(test_losses, label='Test Loss', linestyle='-', color='red')
  ax1.plot(val_losses, label='Validation Loss', linestyle='-.', color='green')
  ax1.set_xlabel('Epochs', fontweight='bold')
  ax1.set_ylabel('Loss', fontweight='bold')
  ax1.set_title('Loss Curves', fontweight='bold')
  ax1.grid(True)
  ax1.legend()

  # Error Curves
  ax2 = plt.subplot(122)
  ax2.plot(train_errors, label='Training Error', linestyle='--', color='blue')
  ax2.plot(test_errors, label='Test Error', linestyle='-', color='red')
  ax2.plot(val_errors, label='Validation Error', linestyle='-.', color='green')
  ax2.set_xlabel('Epochs', fontweight='bold')
  ax2.set_ylabel('Error', fontweight='bold')
  ax2.set_title('0-1 Error Curves', fontweight='bold')
  ax2.grid(True)
  ax2.legend()

  plt.tight_layout()  # Adjust spacing between subplots
  plt.show()

def toy_plot(model, data, y, feature_dict, activation_func, seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  features = data[:, :2].numpy()  # Assuming data is a 2D Torch Tensor
  labels = y.numpy()  # Assuming labels are in the third column

  x_min, x_max = features[:, 0].min(), features[:, 0].max()
  y_min, y_max = features[:, 1].min(), features[:, 1].max()

  x_min = (x_min * 1.1) if x_min < 0 else (x_min * 0.9)
  x_max = (x_max * 1.1) if x_max > 0 else (x_max * 0.9)
  y_min = (y_min * 1.1) if y_min < 0 else (y_min * 0.9)
  y_max = (y_max * 1.1) if y_max > 0 else (y_max * 0.9)

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

  if data.shape[1] > 2:
    # Add noise to the grid
    noise = np.random.randn(grid.shape[0], data.shape[1] - 2)
    # noise = noise / np.linalg.norm(noise, 2, 1, keepdims=True)
    noise = feature_dict['noise_multiplier'] * noise
    grid = torch.cat([grid, torch.tensor(noise)], 1)

  with torch.no_grad():
    model.to('cpu')
    z = model(grid.to(torch.float32)).numpy().reshape(xx.shape)  # Predict scores on the grid

  plt.figure(figsize=(8, 6))

  # Plot the decision boundary with a color gradient
  plt.contourf(xx, yy, z, cmap='coolwarm', alpha=0.5)  # Adjust alpha for transparency

  # Plot the data points with appropriate colors
  plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm', edgecolors='k')

  # Add labels and title
  plt.xlabel("Core Feature")
  plt.ylabel("Spurious Feature")
  plt.title(f"Toy data with learned class decision boundary ({activation_func})")

  plt.show()

def plot_rank(se, le, xlabel, ylabel, title):
  # Set modern style for plot elements
  plt.style.use("seaborn-v0_8-pastel")  # Or choose a different style you prefer

  # Create the plot
  plt.figure(figsize=(5, 5))

  # Plot the curves with different line styles and colors
  plt.plot(le, se, label=title, linestyle='solid', color='red')

  # Customize axes, title, and background
  plt.xlabel(xlabel, fontweight='light')
  plt.ylabel(ylabel, fontweight='light')
  plt.grid(True)  # Turn on grid
  plt.gca().set_facecolor('lightgray')  # Set plot background color

  plt.legend()
  plt.show()

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

def compute_layer_rank(model, activation_func, type, scale = False, input=None):
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