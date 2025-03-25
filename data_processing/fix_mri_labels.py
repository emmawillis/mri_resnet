import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_gg_from_folder_name(folder_name):
    """
    Given a folder name, tries to parse out the -GG# suffix.
    Returns the grade group string (e.g. 'GG1') or None if not found.
    """
    parts = folder_name.rsplit('-GG', 1)
    if len(parts) != 2:
        return None
    # 'parts[1]' might be '1', '2', etc. Reattach the 'GG'
    return 'GG' + parts[1]

def get_folder_grade_counts(base_dir):
    """
    Walks through all folders in base_dir, parses out the
    'GG#' from the folder name, and tallies them.
    Returns a dictionary: { 'GG1': count, 'GG2': count, ... }.
    """
    counts = {}
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        gg = parse_gg_from_folder_name(folder_name)
        if gg is None:
            # Not a recognized pattern, skip or handle differently
            continue

        counts[gg] = counts.get(gg, 0) + 1
    return counts

def plot_and_save_distribution(grade_counts, title, output_path):
    """
    Given a dict of grade_counts (e.g. {'GG1': 10, 'GG2': 5, ...}),
    create a bar chart and save to output_path as a PNG.
    """
    # Sort grade groups in ascending order if you like
    # so that 'GG1'..'GG5' are in natural order
    sorted_keys = sorted(grade_counts.keys(), key=lambda x: int(x.replace('GG', '')))
    sorted_vals = [grade_counts[k] for k in sorted_keys]

    plt.figure(figsize=(6, 4))
    plt.bar(sorted_keys, sorted_vals)  # no custom colors as requested
    plt.title(title)
    plt.xlabel("Grade Group")
    plt.ylabel("Count")
    # Optionally add numeric labels above the bars
    for i, val in enumerate(sorted_vals):
        plt.text(i, val + 0.05, str(val), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def rename_mri_folders(base_dir, uid_to_highest_gg):
    """
    Renames folders in base_dir from e.g.:
      <somePrefix>MR_<seriesUID>-GG1
    to
      <somePrefix>MR_<seriesUID>-GG<n>
    using the 'uid_to_highest_gg' lookup dict for <seriesUID> -> e.g. 'GG4'.
    """

    for folder_name in os.listdir(base_dir):
        old_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(old_path):
            continue

        # 1) Split off the old '-GGX'
        parts = folder_name.rsplit('-GG', 1)
        if len(parts) != 2:
            # If it doesn't match the pattern, skip or handle differently
            continue
        
        left_part, old_grade_sfx = parts  # e.g. "0028-MR_..." and "1"
        
        # 2) Extract the Series UID from inside left_part after "MR_"
        if "MR_" not in left_part:
            # If it doesn't contain 'MR_', skip or handle differently
            continue
        
        prefix, series_uid = left_part.split("MR_", 1)
        
        # 3) Look up the new highest grade for that Series UID
        if series_uid not in uid_to_highest_gg:
            print(f" WARNING: No highest_gg found for {series_uid}, skipping rename. OLD LABEL = {old_grade_sfx}")
            continue
        
        new_grade = uid_to_highest_gg[series_uid]  # e.g. 'GG4'
        
        # 4) Build the new folder name
        new_folder_name = f"{prefix}MR_{series_uid}-{new_grade}"
        new_path = os.path.join(base_dir, new_folder_name)
        
        # 5) Perform the rename if possible
        if os.path.exists(new_path):
            print(f" WARNING: Cannot rename {old_path} to {new_folder_name}, already exists. Skipping.")
            continue
        
        print(f"Renaming:\n  {old_path}\n  => {new_path}\n")
        os.rename(old_path, new_path)

def main():
    # 1) Read your “Highest Grade Per MRI” sheet
    input_xlsx = "MRI_Gleason_Output.xlsx"
    sheet_name = "Highest Grade Per MRI"

    df = pd.read_excel(input_xlsx, sheet_name=sheet_name)

    # 2) Build a dictionary:  seriesUID -> HighestGradeGroup
    uid_to_highest_gg = dict(zip(df["SeriesInstanceUID"], df["HighestGradeGroup"]))

    # 3) Rename folders in train/ and test/
    train_dir = "dataset/train"
    test_dir  = "dataset/test"
    
    rename_mri_folders(train_dir, uid_to_highest_gg)
    rename_mri_folders(test_dir, uid_to_highest_gg)
    
    print("Folder renaming complete!")

    # 4) Now check the final label distribution in train and test
    print("\n=== Checking final distribution in TRAIN ===")
    train_counts = get_folder_grade_counts(train_dir)
    print("Label counts:", train_counts)
    plot_and_save_distribution(train_counts, "Train Label Distribution", "train_label_distribution.png")

    print("\n=== Checking final distribution in TEST ===")
    test_counts = get_folder_grade_counts(test_dir)
    print("Label counts:", test_counts)
    plot_and_save_distribution(test_counts, "Test Label Distribution", "test_label_distribution.png")

    print("\nDone! Distribution plots saved.")

if __name__ == "__main__":
    main()
