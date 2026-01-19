import pandas as pd
import os


def extract_average_row_and_header(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = None
    average_row = None

    for i, line in enumerate(lines):
        if "Discontinuity Time (s)" in line:
            header = line.strip().split(",")
        if line.strip().startswith("Average"):
            average_row = line.strip().split(",")
            break

    return header, average_row


def build_file_paths(base_dir, experiment, categories=None):
    if categories is None:
        categories = ['areas', 'current', 'pH', 'stark']

    base_path = os.path.join(
        base_dir, experiment,
        'Reconstruction_based_on_CO_peak_in_eigenspectra',
        'Interfacial_layer', 'PC1_discontinuities'
    )

    file_paths = {}
    for category in categories:
        category_path = os.path.join(base_path, category)
        file_paths[f'file_path_{category}_dtw'] = os.path.join(category_path, 'DTW_Analysis_Report.csv')
        file_paths[f'file_path_{category}_spearman'] = os.path.join(category_path, 'Spearman_Correlation_Report.csv')

    return file_paths


def process_experiment_reports(experiment, file_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    file_pairs = {
        "areas": ('file_path_areas_dtw', 'file_path_areas_spearman'),
        "current": ('file_path_current_dtw', 'file_path_current_spearman'),
        "pH": ('file_path_pH_dtw', 'file_path_pH_spearman'),
        "stark": ('file_path_stark_dtw', 'file_path_stark_spearman')
    }

    results = {}

    for category, (dtw_key, spearman_key) in file_pairs.items():
        combined_rows = []
        print(f"\nProcessing {experiment} - {category}")

        for file_key in [dtw_key, spearman_key]:
            file_path = file_paths.get(file_key)
            if file_path and os.path.exists(file_path):
                print(f"Found file: {file_path}")
                header, average_row = extract_average_row_and_header(file_path)
                if average_row:
                    print(f"'Average' row found in {file_key}, adding to dataset.")
                    df = pd.DataFrame([average_row])
                    df.insert(0, 'Source', os.path.basename(file_path))
                    df.insert(0, 'Experiment', experiment)
                    combined_rows.append(df)
                else:
                    print(f"No 'Average' row found in: {file_path}")

        if combined_rows:
            final_df = pd.concat(combined_rows, ignore_index=True)
            if header:
                column_names = ["Experiment", "Source"] + header
            else:
                column_names = ["Experiment", "Source"] + [f"Column_{i}" for i in range(final_df.shape[1] - 2)]
            final_df.columns = column_names

            output_file = os.path.join(output_folder, f"{experiment}_{category}_Averages.csv")
            final_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")

            results[category] = final_df
        else:
            print(f"No valid data found for {category} in experiment {experiment}. Skipping save.")

    return results


def combine_all_experiments(all_experiments_data, output_dir):
    combined_files = {}

    for category, dfs in all_experiments_data.items():
        if dfs:
            combined_experiments_df = pd.concat(dfs, ignore_index=True)
            combined_output_file = os.path.join(output_dir, f"All_Experiments_{category}_Averages.csv")
            combined_experiments_df.to_csv(combined_output_file, index=False)
            print(f"\nAll experiments concatenated for {category} and saved to {combined_output_file}")
            combined_files[category] = combined_output_file
        else:
            print(f"No data found for {category}. No combined file created.")

    return combined_files
