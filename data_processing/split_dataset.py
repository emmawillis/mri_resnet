import os
import shutil
import random
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Paths
original_data_dir = r"../dataset/train"
output_dir = r"../dataset"
test_dir = os.path.join(output_dir, "test")

# Create train/test directories
os.makedirs(test_dir, exist_ok=True)

# Group folders by patient ID
patient_to_mris = defaultdict(list)
for folder in os.listdir(original_data_dir):
    if not os.path.isdir(os.path.join(original_data_dir, folder)):
        continue
    patient_id = folder.split("-")[0]  # Extract '0001' from '0001-MR_xxx-GG3'
    patient_to_mris[patient_id].append(folder)

# Split patients
all_patients = list(patient_to_mris.keys())
print(f"Total patients: {len(all_patients)}")
random.shuffle(all_patients)

split_idx = int(len(all_patients) * 0.8)
train_patients = set(all_patients[:split_idx])
test_patients = set(all_patients[split_idx:])

# Move folders
def move_patient_mris(patient_ids, dest_dir):
    for pid in patient_ids:
        for folder in patient_to_mris[pid]:
            src = os.path.join(original_data_dir, folder)
            dst = os.path.join(dest_dir, folder)
            shutil.move(src, dst)
            print(f"Moved {folder} → {dest_dir}")

move_patient_mris(test_patients, test_dir)

print("✅ Patient-level dataset split complete!")
