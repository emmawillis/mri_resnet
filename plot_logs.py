import re
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot train and validation loss from log.")
parser.add_argument("log", type=str, help="Path to the log file")
parser.add_argument("--save", type=str, default=None, help="Path to save the plot (e.g., loss_plot.png)")
args = parser.parse_args()

train_losses_dict = {}
val_losses_dict = {}

with open(args.log, "r") as f:
    for line in f:
        # Ignore "new best model" lines
        if "New best model saved" in line:
            continue

        full_match = re.search(r"Epoch (\d+): Train Loss=([\d.]+), Val Loss=([\d.]+)", line)
        train_only_match = re.search(r"Epoch (\d+): Train Loss=([\d.]+)", line)

        if full_match:
            epoch = int(full_match.group(1))
            train_loss = float(full_match.group(2))
            val_loss = float(full_match.group(3))
            train_losses_dict[epoch] = train_loss
            val_losses_dict[epoch] = val_loss
        elif train_only_match:
            epoch = int(train_only_match.group(1))
            train_loss = float(train_only_match.group(2))
            train_losses_dict[epoch] = train_loss

# Sort by epoch
train_epochs = sorted(train_losses_dict.keys())
train_losses = [train_losses_dict[e] for e in train_epochs]

val_epochs = sorted(val_losses_dict.keys())
val_losses = [val_losses_dict[e] for e in val_epochs]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_losses, marker='o', label='Train Loss', linewidth=2)
plt.plot(val_epochs, val_losses, marker='x', label='Val Loss', linewidth=2)

plt.title("Train vs Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

if args.save:
    plt.savefig(args.save)
    print(f"Plot saved to: {args.save}")
else:
    plt.show()
