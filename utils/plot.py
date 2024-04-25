from lib import *

# Plot loss and error curves
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

# Plot decision boundary
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
    model.eval()
    logits = model(grid.to(torch.float32))
    probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
    predicted_class = torch.argmax(probabilities, dim=1)  # Get the index of the max probability

    z = predicted_class.numpy().reshape(xx.shape)
    # z = (torch.sigmoid(model(grid.to(torch.float32)))).numpy().reshape(xx.shape)  # Predict scores on the grid

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

# Plot layer rank
def plot_rank(se, le, xlabel, ylabel, title):
  # Set modern style for plot elements
  plt.style.use("seaborn-v0_8-pastel")  # Or choose a different style you prefer

  # Create the plot
  plt.figure(figsize=(5, 5))

  # Plot the curves with different line styles and colors
  plt.plot(le, se, label=title, linestyle='solid', color='red', marker='o')

  # Customize axes, title, and background
  plt.xlabel(xlabel, fontweight='light')
  plt.ylabel(ylabel, fontweight='light')
  plt.grid(True)  # Turn on grid
  plt.gca().set_facecolor('lightgray')  # Set plot background color
  plt.locator_params(axis='x', integer=True)
  plt.legend()
  plt.show()

def custom_imshow(axis, ys, args, center=False):
    if center:
        minmax = torch.max(torch.abs(ys)).item()
        axis.imshow(ys.cpu().reshape(args.test_resolution, args.test_resolution).t(),
                    interpolation=None, cmap=args.cmap, extent=(-1, 1, -1, 1), alpha=1,
                    origin='lower', vmin=-minmax, vmax=minmax)
    else:
        axis.imshow(ys.cpu().reshape(args.test_resolution, args.test_resolution).t(),
                    interpolation=None, cmap=args.cmap, extent=(-1, 1, -1, 1), alpha=1,
                    origin='lower')
