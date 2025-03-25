import pandas as pd
import numpy as np

def compute_grade_group(primary, secondary):
    """
    Return GG1..GG5 based on the Gleason table:
      Gleason ≤ 6  -> GG1
      Gleason 7 (3+4) -> GG2
      Gleason 7 (4+3) -> GG3
      Gleason 8    -> GG4
      Gleason 9-10 -> GG5
    """
    total = primary + secondary
    if total <= 6:
        return "GG1"
    elif total == 7:
        # Distinguish 3+4 vs 4+3
        if primary == 3 and secondary == 4:
            return "GG2"
        else:
            return "GG3"
    elif total == 8:
        return "GG4"
    else:
        return "GG5"

# A helper to order the GG strings numerically
grade_order = {"GG1": 1, "GG2": 2, "GG3": 3, "GG4": 4, "GG5": 5}

def main():
    # Input and output paths
    input_csv = "TCIA-Biopsy-Data_2020-07-14.csv"
    output_xlsx = "MRI_Gleason_Output.xlsx"

    # Adjust these to match your CSV columns
    col_series_uid = "Series Instance UID (MRI)"
    col_patient_number = "Patient Number"
    col_primary_gleason = "Primary Gleason"
    col_secondary_gleason = "Secondary Gleason"

    df = pd.read_csv(input_csv)

    output_rows = []
    omitted_count = 0

    for _, row in df.iterrows():
        series_uid = row.get(col_series_uid, np.nan)
        pat_number = row.get(col_patient_number, "")
        primary = row.get(col_primary_gleason, np.nan)
        secondary = row.get(col_secondary_gleason, np.nan)

        if pd.isna(series_uid):
            # Skip if no Series UID
            continue
        
        # Extract patient ID from something like "Prostate-MRI-US-Biopsy-1234"
        patient_id = pat_number.replace("Prostate-MRI-US-Biopsy-", "").strip()

        # If Gleason data is missing, skip (omitted)
        if pd.isna(primary) or pd.isna(secondary):
            omitted_count += 1
            continue

        # Make sure they are integers
        try:
            primary = int(primary)
            secondary = int(secondary)
        except ValueError:
            omitted_count += 1
            continue

        grade_group = compute_grade_group(primary, secondary)

        output_rows.append({
            "SeriesInstanceUID": series_uid,
            "PatientID": patient_id,
            "PrimaryGleason": primary,
            "SecondaryGleason": secondary,
            "GradeGroup": grade_group
        })

    # -----------------------
    # SHEET 1: All valid rows
    # -----------------------
    sheet1_df = pd.DataFrame(output_rows)

    # --------------------------------------------------
    # SHEET 2: Highest grade per MRI (and counts by grade)
    # --------------------------------------------------
    if not sheet1_df.empty:
        # Convert “GradeGroup” to numeric rank
        sheet1_df["grade_numeric"] = sheet1_df["GradeGroup"].map(grade_order)
        
        # Group by SeriesInstanceUID => pick the highest grade found
        highest_by_uid = (
            sheet1_df
            .groupby("SeriesInstanceUID", as_index=False)
            .agg({
                "PatientID": "first",   # or e.g. list if you want all
                "grade_numeric": "max"
            })
        )
        # Map back to GG strings
        highest_by_uid["HighestGradeGroup"] = highest_by_uid["grade_numeric"].map(
            {v: k for k, v in grade_order.items()}
        )
        highest_by_uid.drop(columns=["grade_numeric"], inplace=True)

        # Also compute how many MRIs of each grade (based on “HighestGradeGroup”)
        grade_counts = (
            highest_by_uid["HighestGradeGroup"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "HighestGradeGroup", "HighestGradeGroup": "Count"})
        )
    else:
        highest_by_uid = pd.DataFrame(columns=["SeriesInstanceUID", "PatientID", "HighestGradeGroup"])
        grade_counts = pd.DataFrame(columns=["HighestGradeGroup", "Count"])

    # Write out to Excel
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        # Sheet1: All Biopsy Rows
        sheet1_df.drop(columns="grade_numeric", errors="ignore").to_excel(
            writer, index=False, sheet_name="All Biopsies"
        )
        
        # Sheet2: Highest grade per MRI
        highest_by_uid.to_excel(writer, index=False, sheet_name="Highest Grade Per MRI")

        # Now append the counts of each grade below that table
        start_row = len(highest_by_uid) + 2  # +2 for header & blank line
        grade_counts.to_excel(
            writer,
            sheet_name="Highest Grade Per MRI",
            index=False,
            startrow=start_row
        )

    print(f"Done! Wrote {len(sheet1_df)} valid rows to {output_xlsx}.")
    print(f"Omitted rows (missing Gleason): {omitted_count}")

if __name__ == "__main__":
    main()
