# import os
# from collections import Counter
# import matplotlib.pyplot as plt

# def get_grade_counts(root_dir):
#     """
#     Given a directory (e.g. 'train'), this function returns a Counter
#     mapping grade -> count. It expects subfolders with names ending in '-GG#'.
#     """
#     grade_counts = Counter()
    
#     # List all subfolders in root_dir
#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
        
#         # Make sure we're only looking at directories
#         if os.path.isdir(folder_path):
#             # Extract the last piece of the name after the final '-' (e.g. 'GG1')
#             # Or, if your folder names might have different formats, adapt accordingly
#             parts = folder_name.split('-')
#             if parts:
#                 last_part = parts[-1]
#                 # We expect something like "GG1", "GG2", etc.
#                 if last_part.startswith('GG'):
#                     grade_counts[last_part] += 1
#                 else:
#                     # If the folder doesn't end with GG#, skip or handle differently
#                     pass
#     return grade_counts

# def plot_grade_distribution(grade_counts, output_path, title):
#     """
#     Given a Counter of grade -> count, plots a bar chart and saves it to output_path.
#     """
#     grades = list(grade_counts.keys())
#     counts = [grade_counts[g] for g in grades]
    
#     plt.figure(figsize=(6, 4))
#     plt.bar(grades, counts, color='skyblue')
#     plt.title(title)
#     plt.xlabel('Grade')
#     plt.ylabel('Count')
    
#     # Optionally add labels on top of bars
#     for i, count in enumerate(counts):
#         plt.text(i, count + 0.05, str(count), ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

# if __name__ == "__main__":
#     # Paths to your train and test directories
#     train_dir = "dataset/train"
#     test_dir = "dataset/test"

#     # Count the grades in train
#     train_counts = get_grade_counts(train_dir)
#     # Plot and save the distribution
#     plot_grade_distribution(
#         train_counts, 
#         "train_label_distribution.png", 
#         "Train Grade Distribution"
#     )

#     # Count the grades in test
#     test_counts = get_grade_counts(test_dir)
#     # Plot and save the distribution
#     plot_grade_distribution(
#         test_counts, 
#         "test_label_distribution.png", 
#         "Test Grade Distribution"
#     )

#     print("Done! Saved distributions to train_label_distribution.png and test_label_distribution.png.")


import os
import pandas as pd
from collections import defaultdict

def parse_series_uid_from_folder(folder_name):
    """
    Attempts to parse the SeriesInstanceUID from a folder name
    in the format: <patientID>-MR_<SeriesInstanceUID>-GG#.
    Returns the UID if parsing is successful, or None if not.
    """
    # 1) Split off the trailing '-GG#' (unknown #)
    parts = folder_name.rsplit('-GG', 1)
    if len(parts) < 2:
        return None  # doesn't match expected pattern
    
    left_part = parts[0]  # e.g. "<patientID>-MR_<SeriesInstanceUID>"
    
    # 2) Split on 'MR_'
    if 'MR_' not in left_part:
        return None
    prefix, series_uid = left_part.split('MR_', 1)
    
    if not series_uid.strip():
        return None
    
    return series_uid.strip()

def check_dataset_folders(excel_path, sheet_name, dataset_dir):
    """
    1. Reads the Excel file's "Highest Grade Per MRI" sheet 
       (with columns [SeriesInstanceUID, HighestGradeGroup]).
    2. Builds a dictionary of series_uid -> HighestGradeGroup.
    3. Iterates over subfolders in dataset_dir:
       - parse the SeriesUID,
       - check if it's in the dict,
       - track matched/unmatched,
       - if matched, increment a count for that grade group.
    4. Returns (count_matched, count_unmatched, grade_counts_dict).
    """
    # 1) Read the sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # 2) Build a dictionary from SeriesInstanceUID -> HighestGradeGroup
    #    Ensure SeriesInstanceUID is a string
    uid_to_gg = dict(zip(df["SeriesInstanceUID"].astype(str), df["HighestGradeGroup"]))
    
    # 3) Traverse folders
    count_matched = 0
    count_unmatched = 0
    grade_counts = defaultdict(int)

    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # skip files

        uid = parse_series_uid_from_folder(folder_name)
        if uid is None:
            # Could not parse -> unmatched
            count_unmatched += 1
            continue
        
        # Check dictionary for a match
        if uid in uid_to_gg:
            count_matched += 1
            # Increment the count for that grade
            gg = uid_to_gg[uid]
            grade_counts[gg] += 1
        else:
            count_unmatched += 1

    return count_matched, count_unmatched, grade_counts

def print_grade_counts(grade_counts):
    """
    Prints the counts of each grade group in a nice format.
    """
    if not grade_counts:
        print("  (No matched folders or no grade data.)")
        return
    
    # Sort by GradeGroup name (GG1..GG5), or just alphabetically
    for gg in sorted(grade_counts.keys()):
        print(f"    {gg}: {grade_counts[gg]}")

def main():
    excel_path = "MRI_Gleason_Output.xlsx"
    sheet_name = "Highest Grade Per MRI"

    train_dir = "dataset/train"
    test_dir  = "dataset/test"

    print("=== TRAIN FOLDERS ===")
    train_matched, train_unmatched, train_grade_counts = check_dataset_folders(
        excel_path, sheet_name, train_dir
    )
    print(f"  Matched:   {train_matched}")
    print(f"  Unmatched: {train_unmatched}")
    print("  Grade distribution for matched folders:")
    print_grade_counts(train_grade_counts)

    print("\n=== TEST FOLDERS ===")
    test_matched, test_unmatched, test_grade_counts = check_dataset_folders(
        excel_path, sheet_name, test_dir
    )
    print(f"  Matched:   {test_matched}")
    print(f"  Unmatched: {test_unmatched}")
    print("  Grade distribution for matched folders:")
    print_grade_counts(test_grade_counts)

    print("\nDone checking dataset folders!")

if __name__ == "__main__":
    main()
