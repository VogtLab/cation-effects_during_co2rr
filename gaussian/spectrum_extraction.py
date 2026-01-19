import os
import pandas as pd
import numpy as np


def find_integrated_area_files(base_path, ignore_folders=None):
    if ignore_folders is None:
        ignore_folders = set()

    ignore_folders_lower = {f.lower() for f in ignore_folders}

    integrated_area_files = []
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d.lower() not in ignore_folders_lower]
        for file in files:
            if file.endswith("_integrated_areas.csv"):
                integrated_area_files.append(os.path.join(root, file))

    return integrated_area_files


def extract_integrated_areas_at_time(base_dir, include_folders, target_peaks, target_time,
                                      ignore_folders=None, tolerance=0.1):
    if ignore_folders is None:
        ignore_folders = {
            "raw data", "1635_peak", "CO peak", "900_to_3999",
            "650_to_4000", "2000_to_3999", "non-mean-centered", "Stark shift"
        }

    results = []
    processed_files = set()

    for folder_name in include_folders:
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        integrated_files = find_integrated_area_files(folder_path, ignore_folders)
        if not integrated_files:
            print(f"No integrated areas files found in {folder_path}")
            continue

        for full_path in integrated_files:
            file_key = (os.path.basename(full_path), os.path.dirname(full_path))
            if file_key in processed_files:
                print(f"Skipping duplicate file: {full_path}")
                continue
            processed_files.add(file_key)

            print(f"Processing file: {full_path}")
            df_integrated = pd.read_csv(full_path)

            if 'Time (s)' not in df_integrated.columns:
                print(f"'Time (s)' column not found in {full_path}")
                continue

            row = df_integrated[np.isclose(df_integrated['Time (s)'], target_time, atol=tolerance)]
            if not row.empty:
                experiment_group = os.path.relpath(full_path, base_dir).split(os.sep)[0]

                row_data = {
                    'Experiment': experiment_group,
                    'File': os.path.basename(full_path),
                    'Folder': os.path.dirname(full_path),
                }

                for peak in target_peaks:
                    peak_col = f'Mean {peak}'
                    row_data[peak_col] = row[peak_col].values[0] if peak_col in row else None

                results.append(row_data)
            else:
                print(f"Target time {target_time} not found in {full_path}")

    return results


def save_consolidated_results(results, output_path):
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.drop_duplicates()
        results_df.to_csv(output_path, index=False)
        print(f"Consolidated results saved to: {output_path}")
        return results_df
    else:
        print("No data to consolidate. Results list is empty.")
        return None
