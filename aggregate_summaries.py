import os
import pandas as pd

# Define the main folder
exp_folder = "test_squeezenet"

# List to store DataFrames
dataframes = []

# Loop through each subfolder in the main folder
for subfolder in os.listdir(exp_folder):
    subfolder_path = os.path.join(exp_folder, subfolder)
    # Check if it's a directory and contains summary.csv
    if os.path.isdir(subfolder_path):
        summary_file = os.path.join(subfolder_path, "summary.csv")
        if os.path.exists(summary_file):
            # Read the summary.csv and add a column for subfolder name
            df = pd.read_csv(summary_file)
            df["Experiment"] = subfolder  # Add experiment identifier
            dataframes.append(df)

# Merge all DataFrames
if dataframes:
    all_summary = pd.concat(dataframes, ignore_index=True)
    # Save to all_summary.csv in the main folder
    all_summary_path = os.path.join(exp_folder, "all_summary.csv")
    all_summary.to_csv(all_summary_path, index=False)
    print(f"All summaries merged and saved to {all_summary_path}")
else:
    print("No summary.csv files found in the subfolders.")