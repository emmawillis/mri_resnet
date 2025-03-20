import os
import shutil
import pandas as pd

# Define paths
source_dir = r"P:\data\prostate_mri_us_biopsy\images"  # Source MRI folder
dest_dir = r"P:\data\prostate_mri_us_biopsy\60channelMRIs"  # Destination folder
excel_file = r"P:\data\prostate_mri_us_biopsy\TCIA Biopsy Data_2020-07-14.xlsx"  # Path to Excel file

# Ensure destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Load the Excel file
df = pd.read_excel(excel_file)

# Extract relevant columns
df = df[['Primary Gleason', 'Secondary Gleason', 'Patient Number']]

# Remove any missing values
df = df.dropna()

# Convert Gleason Scores to integers
df['Primary Gleason'] = df['Primary Gleason'].astype(int)
df['Secondary Gleason'] = df['Secondary Gleason'].astype(int)

# Compute total Gleason Score and map it to a Gleason Grade
df['Gleason Score'] = df['Primary Gleason'] + df['Secondary Gleason']

# Define mapping of Gleason Score to Grade Group
def map_gleason_to_grade(score):
    if score <= 6:
        return 1
    elif score == 7 and df['Primary Gleason'].eq(3).all():
        return 2  # Gleason 3+4
    elif score == 7 and df['Primary Gleason'].eq(4).all():
        return 3  # Gleason 4+3
    elif score == 8:
        return 4
    elif score >= 9:
        return 5
    return 1

df['Gleason Grade'] = df.apply(lambda row: map_gleason_to_grade(row['Gleason Score']), axis=1)

# Get the highest Gleason Grade per patient
max_gleason_per_patient = df.groupby('Patient Number')['Gleason Grade'].max().to_dict()

# Process MRI files
for patient_folder in os.listdir(source_dir):
    patient_path = os.path.join(source_dir, patient_folder)
    
    if os.path.isdir(patient_path):  # Ensure it's a folder
        patient_number = patient_folder.split('-')[-1]  # Extract patient number
        
        # Get highest Gleason Grade for this patient
        highest_gleason_grade = max_gleason_per_patient.get(patient_folder, None)
        
        if highest_gleason_grade is None:
            print(f"Skipping {patient_folder}: No biopsy data found.")
            continue  # Skip patients with no biopsy data
        
        # Iterate through MRI folders inside each patient folder
        for mri_folder in os.listdir(patient_path):
            mri_path = os.path.join(patient_path, mri_folder)
            
            if os.path.isdir(mri_path):  # Ensure it's a folder
                dcm_files = [f for f in os.listdir(mri_path) if f.endswith('.dcm')]

                # Check if this MRI folder contains exactly 60 .dcm files
                if len(dcm_files) == 60:
                    new_folder_name = f"{patient_number}-{mri_folder}-GG{highest_gleason_grade}"
                    new_folder_path = os.path.join(dest_dir, new_folder_name)

                    # Copy the entire MRI folder
                    shutil.copytree(mri_path, new_folder_path)
                    print(f"Copied {mri_path} to {new_folder_path}")

print("Processing complete!")
