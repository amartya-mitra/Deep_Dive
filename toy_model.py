import torch
import torch.nn as nn
import torch.optim as optim

from lib import *
from plot import *


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
    
def train_model(model, epochs, use_es, use_gpu, train_dict, inputs, target, seed=0):
  # Set seed
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

  # Early stopping parameters
  no_improve_epochs = 0
  min_loss_change = train_dict['min_loss_change']  # Define 'substantial' change here
  no_improve_threshold = train_dict['no_improve_threshold']  # Stop if no improvement in 5 epochs
  best_loss = float('inf')

  # Binary Cross Entropy Loss for binary classification
  criterion = nn.BCELoss()

  # Optimizer
  optimizer = optim.SGD(model.parameters(),
                        lr=train_dict['lr'],
                        momentum=train_dict['momentum'])

  # For plotting metrics
  train_losses, test_losses, val_losses = [], [], []
  train_errors, test_errors, val_errors = [], [], []

  train_size = math.floor(len(inputs)*0.6)
  test_size = math.floor(len(inputs)*0.2)
  val_size = len(inputs) - train_size - test_size

  X_train, y_train = inputs[:train_size].to(device), target[:train_size].to(device)
  X_test, y_test = inputs[train_size:train_size+test_size].to(device), target[train_size:train_size+test_size].to(device)
  X_val, y_val = inputs[train_size+test_size:].to(device), target[train_size+test_size:].to(device)

  # model =
  model.to(device)

  # Calculate total steps for tqdm
  total_steps = epochs

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
      train_predictions = torch.round(train_outputs)  # Convert probabilities to 0/1 predictions
      train_error = torch.mean((train_predictions != (y_train + 1)/2).float())

      test_outputs = model(X_test)
      test_loss = train_dict['loss_mp'] * criterion(test_outputs, (y_test + 1)/2)
      test_predictions = torch.round(test_outputs)  # Convert probabilities to 0/1 predictions
      test_error = torch.mean((test_predictions != (y_test + 1)/2).float())

      val_outputs = model(X_val)
      val_loss = train_dict['loss_mp'] * criterion(val_outputs, (y_val + 1)/2)
      val_predictions = torch.round(val_outputs)  # Convert probabilities to 0/1 predictions
      val_error = torch.mean((val_predictions != (y_val + 1)/2).float())

      ################## Append Metrics ###############

      train_losses.append(train_loss.item())
      train_errors.append(train_error.item())

      test_losses.append(test_loss.item())
      test_errors.append(test_error.item())

      val_losses.append(val_loss.item())
      val_errors.append(val_error.item())

      if use_es:

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

      # Update tqdm bar
      pbar.update(1)
      pbar.set_postfix_str(f'Loss: {loss.item():.4f}, 0-1 Error: {train_error.item():.4f}')

  # Plot the metrics
  plot_metrics(train_losses, test_losses, val_losses, train_errors, test_errors, val_errors)

  # The model is now trained on the dataset with two features
  # and the associated binary labels using a single output node and BCE loss
  return model