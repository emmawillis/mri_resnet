import re
import argparse
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description="Plot train and validation loss from log.")
parser.add_argument("log", type=str, help="Path to the log file")
parser.add_argument("--save", type=str, default=None, help="Path to save the plot (e.g., loss_plot.png)")
args = parser.parse_args()

# Extract loss values
epochs = []
train_losses = []
val_losses = []

with open(args.log, "r") as f:
    for line in f:
        match = re.search(r"Epoch (\d+): Train Loss=([\d.]+), Val Loss=([\d.]+)", line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2)
plt.plot(epochs, val_losses, marker='x', label='Val Loss', linewidth=2)
plt.title("Train vs Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save or show
if args.save:
    plt.savefig(args.save)
    print(f"Plot saved to: {args.save}")
else:
    plt.show()
