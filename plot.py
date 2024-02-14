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