import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pydicom
import torchio as tio
from pytorchvideo.models.resnet import create_resnet
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset

# Label mapping for multi-class (GG1 to GG5)
gleason_labels = {"GG1": 0, "GG2": 1, "GG3": 2, "GG4": 3, "GG5": 4}
class_names = list(gleason_labels.keys())

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, binary=False):
        self.root_dir = root_dir
        self.transform = transform
        self.binary = binary
        self.samples = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                label_str = folder_name.split("-")[-1]
                if self.binary:
                    # Only keep low (GG1) and high (GG4 or GG5) samples
                    if label_str == "GG1":
                        label = 0  # low
                        self.samples.append((folder_path, label))
                    elif label_str in ["GG4", "GG5"]:
                        label = 1  # high
                        self.samples.append((folder_path, label))
                    else:
                        # Ignore GG2 and GG3
                        continue
                else:
                    if label_str in gleason_labels:
                        label = gleason_labels[label_str]
                        self.samples.append((folder_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        dicom_slices = []
        for fname in os.listdir(folder_path):
            if fname.endswith(".dcm"):
                fpath = os.path.join(folder_path, fname)
                dcm = pydicom.dcmread(fpath)
                instance_number = getattr(dcm, "InstanceNumber", 0)
                dicom_slices.append((instance_number, dcm))

        dicom_slices.sort(key=lambda x: x[0])
        volume = [
            (dcm.pixel_array.astype(np.float32) - np.min(dcm.pixel_array)) / 
            (np.max(dcm.pixel_array) - np.min(dcm.pixel_array) + 1e-5)
            for _, dcm in dicom_slices
        ]
        volume = torch.tensor(np.stack(volume, axis=0)).unsqueeze(0)

        transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.Resize((60, 256, 256)),
        ])

        subject = tio.Subject(mri=tio.ScalarImage(tensor=volume))
        volume = transform(subject).mri.data

        if self.transform:
            volume = self.transform(volume)

        return volume, label

def train(binary=False):
    # Set number of classes based on mode
    num_classes = 2 if binary else 5

    # Create the ResNet50 model with the correct number of classes.
    model = create_resnet(
        input_channel=1,
        model_depth=50,
        norm=torch.nn.BatchNorm3d,
    )
    model.blocks[5].proj = nn.Linear(in_features=2048, out_features=num_classes)

    model = model.to("cuda")

    print(model)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])

    # Create datasets with the binary flag passed on.
    train_dataset = MRIDataset("./dataset/train", transform=transform, binary=binary)
    val_dataset = MRIDataset("./dataset/test", transform=transform, binary=binary)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)  # Adjust batch size as needed
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

    # Check DataLoader output for one batch.
    for images, labels in train_loader:
        print("Batch Image Shape:", images.shape)  # Expected shape: (Batch, 1, 60, H, W)
        print("Batch Labels:", labels)
        break

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    checkpoint_path = "./best_model.pt"

    for epoch in range(1, 31):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Save checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch}: New best model saved (val_loss={avg_val_loss:.4f})")

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Classification report
        if binary:
            target_names = ["Low", "High"]
            labels_list = [0, 1]
        else:
            target_names = [label for label, _ in sorted(gleason_labels.items(), key=lambda x: x[1])]
            labels_list = [0, 1, 2, 3, 4]

        report = classification_report(
            all_labels,
            all_preds,
            labels=labels_list,
            target_names=target_names,
            zero_division=0
        )
        print("\nClassification Report:\n", report)
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
