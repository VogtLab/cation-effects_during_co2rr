import os
import pandas as pd
import matplotlib as mpl

from alignment_references import create_reference_curves
from alignment_processing import process_pca_file, save_references_to_csv_and_plot

mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'


base_dir = "./data"
output_base_dir = "./alignment_output"

include_folders = {
    "DS_00132", "DS_00133",
}

ignore_folders = {
    "raw data", "CO peak", "Stark shift", "2500_to_3999", "1001_to_3999",
    "1635_peak", "2000_to_3999", "900_to_3999", "650_to_4000",
    "Diffusion coefficient plots", "DS_00145_01", "First derivative",
    "non-mean-centered", "test", "650 to 4000"
}

row_range = (80, 180)
col_range = (1, 16)


if __name__ == "__main__":
    os.makedirs(output_base_dir, exist_ok=True)

    references = create_reference_curves(n_points=100, n_flat=9, step_size=2)

    references_output_dir = os.path.join(output_base_dir, "Reference_Curves")
    save_references_to_csv_and_plot(references, references_output_dir)

    summary_results = []

    for folder_name in include_folders:
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d.lower() not in {f.lower() for f in ignore_folders}]

            for file_name in files:
                if file_name == "PCA_scores.txt":
                    file_path = os.path.join(root, file_name)
                    print(f"Processing file: {file_path}")

                    try:
                        file_output_dir = os.path.join(output_base_dir, folder_name)
                        os.makedirs(file_output_dir, exist_ok=True)

                        results = process_pca_file(
                            file_path, references, file_output_dir,
                            row_range=row_range, col_range=col_range
                        )

                        summary_results.extend(results)

                        results_df = pd.DataFrame(results)
                        results_csv_path = os.path.join(file_output_dir, "DTW_Results.csv")
                        results_df.to_csv(results_csv_path, index=False)
                        print(f"Results saved to: {results_csv_path}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(output_base_dir, "DTW_Summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved to: {summary_csv_path}")

    print("\nProcessing complete.")
