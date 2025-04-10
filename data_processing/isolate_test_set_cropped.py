import os
import shutil
import random
from collections import defaultdict, Counter

# Set seed for reproducibility
random.seed(42)

# Paths
INPUT_DIR = "dataset_cropped"
OUTPUT_DIR = "dataset_cropped"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Create output directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Map patientID -> list of their MRI folders
patient_to_mris = defaultdict(list)

for folder in os.listdir(INPUT_DIR):
    full_path = os.path.join(INPUT_DIR, folder)
    if not os.path.isdir(full_path) or folder in ["train", "test"]:
        continue
    patient_id = folder.split("-MR_")[0]
    patient_to_mris[patient_id].append(folder)

# Log how many MRIs each patient has
mri_counts = Counter(len(mris) for mris in patient_to_mris.values())
total_patients = len(patient_to_mris)

print(f"\nTotal unique patients: {total_patients}")
print("Patients with N MRIs:")
for n, count in sorted(mri_counts.items()):
    print(f"  {n} MRIs: {count} patients")

# Split patients
all_patients = list(patient_to_mris.keys())
random.shuffle(all_patients)

num_test = int(0.2 * total_patients)
test_patients = set(all_patients[:num_test])
train_patients = set(all_patients[num_test:])

print(f"\nTrain patients: {len(train_patients)}")
print(f"Test patients:  {len(test_patients)}")

# Move folders to their splits and collect test class counts
low_count = 0
high_count = 0

for patient_id, folders in patient_to_mris.items():
    split_dir = TEST_DIR if patient_id in test_patients else TRAIN_DIR
    for folder in folders:
        src = os.path.join(INPUT_DIR, folder)
        dst = os.path.join(split_dir, folder)
        shutil.copytree(src, dst)

        if split_dir == TEST_DIR:
            if "-GG1" in folder:
                low_count += 1
            elif "-GG4" in folder or "-GG5" in folder:
                high_count += 1

print("\nSplit complete.")
print(f"\n[TEST SET CLASS DISTRIBUTION]")
print(f"  GG1 (LOW):  {low_count}")
print(f"  GG4/5 (HIGH): {high_count}")
