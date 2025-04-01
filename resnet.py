import os
import sys
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import torch.nn.functional as F
import pydicom
import torchio as tio
from pytorchvideo.models.resnet import create_resnet
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter

# Label mapping for multi-class (GG1 to GG5)
gleason_labels = {"GG1": 0, "GG2": 1, "GG3": 2, "GG4": 3, "GG5": 4}
class_names = list(gleason_labels.keys())

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # index alpha per target class
        else:
            alpha_factor = 1.0

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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
                    if label_str == "GG1":
                        label = 0  # low
                        self.samples.append((folder_path, label))
                    elif label_str in ["GG4", "GG5"]:
                        label = 1  # high
                        self.samples.append((folder_path, label))
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


def train(binary=False, epochs=300):
    dataset = MRIDataset("./dataset/train", transform=None, binary=binary)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    gamma_values = [0.5, 1.0, 2.0, 3.0]
    best_gamma = None
    best_cv_loss = float('inf')

    os.makedirs("logs", exist_ok=True)

    summary_path = "logs/gamma_summary.csv"
    with open(summary_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Gamma", "Avg_CV_Loss"])

        for gamma in gamma_values:
            log_file_path = f"logs/gamma_{gamma}.log"
            sys.stdout = open(log_file_path, "w")

            print(f"\nTuning with gamma = {gamma}")
            fold_losses = []

            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                print(f"--- Fold {fold + 1} ---")

                train_transform = tio.Compose([
                    tio.RandomFlip(axes=('LR',), flip_probability=0.5),
                    tio.RandomAffine(scales=(0.95, 1.05),degrees=5),
                    tio.RescaleIntensity(out_min_max=(0, 1)),
                    tio.Resize((60, 256, 256)),
                ])

                val_transform = tio.Compose([
                    tio.RescaleIntensity(out_min_max=(0, 1)),
                    tio.Resize((60, 256, 256)),
                ])

                train_subset = Subset(dataset, train_ids)
                val_subset = Subset(dataset, val_ids)
                train_subset.dataset.transform = train_transform
                val_subset.dataset.transform = val_transform

                labels = [label for _, label in [dataset[i] for i in train_ids]]
                class_sample_count = np.array([len(np.where(np.array(labels) == t)[0]) for t in np.unique(labels)])
                weights = 1. / class_sample_count
                samples_weights = np.array([weights[t] for t in labels])
                sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

                train_loader = DataLoader(train_subset, batch_size=4, sampler=sampler, num_workers=0)
                val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0)

                model = create_resnet(
                    input_channel=1,
                    model_depth=50,
                    norm=torch.nn.BatchNorm3d,
                )
                num_classes = 2 if binary else 5
                model.blocks[5].proj = nn.Linear(in_features=2048, out_features=num_classes)
                model = model.to("cuda")

                class_weights = torch.tensor(weights, dtype=torch.float).to("cuda")
                criterion = FocalLoss(alpha=class_weights, gamma=gamma)
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                best_val_loss = float('inf')

                for epoch in range(1, epochs+1):
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

                    if epoch % 5 == 0:
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

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save(model.state_dict(), f"./best_model_fold{fold}_gamma{gamma}.pt")
                            print(f"Epoch {epoch}: New best model saved for fold {fold} (val_loss={avg_val_loss:.4f})")

                        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

                        target_names = ["Low", "High"] if binary else list(gleason_labels.keys())
                        labels_list = [0, 1] if binary else list(range(num_classes))

                        report = classification_report(
                            all_labels,
                            all_preds,
                            labels=labels_list,
                            target_names=target_names,
                            zero_division=0
                        )
                        print("\nClassification Report:\n", report)
                        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
                    else:
                        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}")

                fold_losses.append(best_val_loss)

            avg_cv_loss = sum(fold_losses) / len(fold_losses)
            print(f"\nAverage CV Loss for gamma={gamma}: {avg_cv_loss:.4f}")
            writer.writerow([gamma, avg_cv_loss])

            if avg_cv_loss < best_cv_loss:
                best_cv_loss = avg_cv_loss
                best_gamma = gamma

            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print(f"\nBest gamma: {best_gamma} with Avg CV Loss: {best_cv_loss:.4f}")
