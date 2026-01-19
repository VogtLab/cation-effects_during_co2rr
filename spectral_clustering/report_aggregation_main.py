import os

from report_aggregation import (
    build_file_paths,
    process_experiment_reports,
    combine_all_experiments
)


base_dir = "./data"
output_dir = "./aggregated_reports"

experiments = [
    'DS_00132',
    'DS_00133',
    'DS_00127',
]

categories = ['areas', 'current', 'pH', 'stark']


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    all_experiments_data = {category: [] for category in categories}

    for experiment in sorted(experiments):
        print(f"\n{'='*60}")
        print(f"PROCESSING EXPERIMENT: {experiment}")
        print(f"{'='*60}")

        file_paths = build_file_paths(base_dir, experiment, categories)

        experiment_output_folder = os.path.join(output_dir, experiment, "Averages_Combined")

        results = process_experiment_reports(experiment, file_paths, experiment_output_folder)

        for category, df in results.items():
            all_experiments_data[category].append(df)

    print(f"\n{'='*60}")
    print("COMBINING ALL EXPERIMENTS")
    print(f"{'='*60}")

    combine_all_experiments(all_experiments_data, output_dir)

    print(f"\nProcessing complete. Files saved in {output_dir}")
