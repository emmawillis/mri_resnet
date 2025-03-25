import pandas as pd

# Define input and output file paths
input_excel_file =r"P:\data\prostate_mri_us_biopsy\TCIA Biopsy Data_2020-07-14.xlsx"  # Update this path if needed
output_excel_file = r"P:\data\prostate_mri_us_biopsy\processed_gleason_data.xlsx"

# Load the Excel file
df = pd.read_excel(input_excel_file)

# Extract relevant columns
df = df[['Primary Gleason', 'Secondary Gleason', 'Patient Number']].dropna()

# Convert Gleason Scores to integers
df['Primary Gleason'] = df['Primary Gleason'].astype(int)
df['Secondary Gleason'] = df['Secondary Gleason'].astype(int)

# Compute total Gleason Score
df['Gleason Score'] = df['Primary Gleason'] + df['Secondary Gleason']

# Define mapping of Gleason Score to Grade Group
def map_gleason_to_grade(row):
    if row['Gleason Score'] <= 6:
        return 1
    elif row['Gleason Score'] == 7 and row['Primary Gleason'] == 3:
        return 2  # Gleason 3+4
    elif row['Gleason Score'] == 7 and row['Primary Gleason'] == 4:
        return 3  # Gleason 4+3
    elif row['Gleason Score'] == 8:
        return 4
    elif row['Gleason Score'] >= 9:
        return 5
    return None  # Default to Grade Group 1 (benign) if unexpected

df['Gleason Grade'] = df.apply(map_gleason_to_grade, axis=1)

# Get the highest Gleason Grade per patient
max_gleason_per_patient = df.groupby('Patient Number')['Gleason Grade'].max().reset_index()
max_gleason_per_patient.rename(columns={'Gleason Grade': 'Max Gleason Grade'}, inplace=True)

# Save both sheets to an Excel file
with pd.ExcelWriter(output_excel_file) as writer:
    df.to_excel(writer, sheet_name="Biopsy Gleason Data", index=False)
    max_gleason_per_patient.to_excel(writer, sheet_name="Max Gleason Per Patient", index=False)

print(f"Processed data saved to {output_excel_file}")
