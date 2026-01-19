import pandas as pd
import numpy as np
import os
import traceback

from discontinuity_detection import (
    find_integrated_areas_file,
    analyze_discontinuities,
    get_surrounding_points,
    smooth_signal
)
from discontinuity_correlation import (
    calculate_spearman_for_discontinuities,
    prepare_spearman_matrix,
    calculate_dtw_cost_with_normalization,
    prepare_dtw_matrix
)
from discontinuity_visualization import (
    plot_spearman_visualization,
    plot_spearman_visualization_with_arrows,
    explain_spearman_visualization,
    plot_dtw_matrices_and_path,
    plot_discontinuities,
    plot_spearman_heatmap,
    plot_dtw_heatmap
)


base_dir = "./data"

experiment_configs = [
    {
        'experiment': 'EXPERIMENT_001',
        'window_length': 15,
        'polyorder': 2,
        'peak_threshold': 0.5,
        'trough_threshold': 0.5,
        'magnitude_threshold': 0.4
    },
]

analysis_configurations = [
    {
        'folder_name': 'areas',
        'filename_suffix': 'areas',
        'subfolder': '1101_to_3999',
        'file_ending': 'integrated_areas.csv'
    },
    {
        'folder_name': 'pH',
        'filename_suffix': 'pH',
        'subfolder': '1101_to_3999',
        'file_ending': '1430_1368.csv'
    },
    {
        'folder_name': 'stark',
        'filename_suffix': 'stark',
        'subfolder': 'Stark_shift',
        'file_ending': 'smoothed_integrated_areas.csv'
    }
]

potential_change_times = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

invert_pc1 = True


def save_spearman_report(folder_path, spearman_df, discontinuity_info, metadata_df, discontinuity_times):
    spearman_averages = spearman_df.mean()
    spearman_df.loc['Average'] = spearman_averages

    csv_filename = os.path.join(folder_path, 'Spearman_Correlation_Report.csv')

    with open(csv_filename, 'w') as f:
        metadata_df.to_csv(f, index=False)
        f.write("\n")
        discontinuity_info.to_csv(f, index=False)
        f.write("\n")
        spearman_df.to_csv(f, index_label='Discontinuity Time (s)')

    print(f"Spearman correlation report saved to {csv_filename}")


def save_dtw_report(folder_path, dtw_df, discontinuity_info, metadata_df, discontinuity_times):
    dtw_averages = dtw_df.mean()
    dtw_df.loc['Average'] = dtw_averages

    csv_filename_dtw = os.path.join(folder_path, 'DTW_Analysis_Report.csv')

    with open(csv_filename_dtw, 'w') as f:
        metadata_df.to_csv(f, index=False)
        f.write("\n")
        discontinuity_info.to_csv(f, index=False)
        f.write("\n")
        dtw_df.to_csv(f, index_label='Discontinuity Time (s)')

    print(f"DTW alignment cost report saved to {csv_filename_dtw}")


def process_experiment(exp_config, base_dir, configurations, potential_change_times, invert_pc1=True):
    experiment = exp_config['experiment']
    file_path = os.path.join(base_dir, experiment, 'Reconstruction_based_on_CO_peak_in_eigenspectra',
                             'Interfacial_layer', 'PCA_scores.txt')

    if not os.path.exists(file_path):
        print(f"WARNING: File not found for {experiment}, skipping...")
        return False

    data = pd.read_csv(file_path, delim_whitespace=True, header=None)

    time = data[0].values[:] * 1.1
    pc1 = data[1].values[:]
    if invert_pc1:
        pc1 = pc1 * (-1)

    window_length = exp_config['window_length']
    polyorder = exp_config['polyorder']
    peak_threshold = exp_config['peak_threshold']
    trough_threshold = exp_config['trough_threshold']
    magnitude_threshold = exp_config['magnitude_threshold']

    print(f"Parameters: window_length={window_length}, polyorder={polyorder}")
    print(f"            peak_threshold={peak_threshold}, trough_threshold={trough_threshold}")
    print(f"            magnitude_threshold={magnitude_threshold}")

    pc1_smoothed = smooth_signal(pc1, window_length, polyorder)

    disc_times, disc_magnitudes, disc_values, potential_changes = analyze_discontinuities(
        time, pc1_smoothed,
        peak_threshold=peak_threshold,
        trough_threshold=trough_threshold,
        potential_change_times=potential_change_times,
        magnitude_threshold=magnitude_threshold)

    if len(disc_magnitudes) == 0:
        print(f"\nWARNING: No discontinuities found for {experiment}")
        return False

    print(f"\nDiscontinuities detected: {len(disc_magnitudes)}")
    print(f"Average magnitude: {np.mean(disc_magnitudes):.2f}")
    print(f"Standard deviation: {np.std(disc_magnitudes):.2f}")

    for config in configurations:
        print(f"\n{'-'*60}")
        print(f"Processing configuration: {config['filename_suffix']}")
        print(f"{'-'*60}\n")

        folder = os.path.dirname(file_path)
        folder_path = os.path.join(folder, f'PC1_discontinuities/{config["folder_name"]}')

        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.basename(folder) + f"_PC1_with_discontinuities_{config['filename_suffix']}"

        plot_discontinuities(time, pc1_smoothed, disc_times, disc_magnitudes,
                             potential_changes, folder_path, filename)

        surrounding_times, surrounding_values, surrounding_indices = get_surrounding_points(
            time, pc1_smoothed, disc_times)

        integrated_areas_file = find_integrated_areas_file(
            file_path, config['subfolder'], config['file_ending'])

        if integrated_areas_file is None:
            print(f"WARNING: Could not find integrated areas file for {config['filename_suffix']}, skipping...")
            continue

        integrated_areas_data = pd.read_csv(integrated_areas_file, header=0,
                                            usecols=lambda col: col != 'Time (s)')

        spearman_results = calculate_spearman_for_discontinuities(
            surrounding_values, surrounding_indices, integrated_areas_data)

        output_folder = os.path.join(folder_path, 'Spearman_Visualizations')
        os.makedirs(output_folder, exist_ok=True)

        for i, (pca_values, int_values, time_val) in enumerate(
                zip(surrounding_values, integrated_areas_data.values, disc_times)):
            plot_spearman_visualization(pca_values, int_values, time_val, output_folder)
            plot_spearman_visualization_with_arrows(pca_values, int_values, time_val, output_folder)
            explain_spearman_visualization(pca_values, int_values, time_val, output_folder)

        discontinuity_times = [f"{t:.2f} s" for t in disc_times]
        matrix, discontinuities, peaks = prepare_spearman_matrix(spearman_results)

        plot_spearman_heatmap(matrix, discontinuity_times, peaks, folder_path)

        spearman_df = pd.DataFrame(spearman_results).T
        spearman_df.index = discontinuity_times
        discontinuity_info = pd.DataFrame({
            'Discontinuity Time (s)': discontinuity_times,
            'Discontinuity Magnitude': disc_magnitudes,
            'Discontinuity Value': disc_values
        })

        avg_magnitude = np.mean(disc_magnitudes) if len(disc_magnitudes) > 0 else "No discontinuities"
        std_magnitude = np.std(disc_magnitudes) if len(disc_magnitudes) > 0 else "No discontinuities"

        metadata = {
            'SG Filter Window Length': [window_length],
            'SG Filter Polyorder': [polyorder],
            'Peak Threshold': [peak_threshold],
            'Trough Threshold': [trough_threshold],
            'Magnitude Threshold': [magnitude_threshold],
            'Average Discontinuity Magnitude': [avg_magnitude],
            'Standard Deviation of Discontinuity Magnitude': [std_magnitude]
        }

        metadata_df = pd.DataFrame(metadata)

        save_spearman_report(folder_path, spearman_df, discontinuity_info, metadata_df, discontinuity_times)

        output_folder_dtw = os.path.join(folder_path, 'DTW_Plots')
        os.makedirs(output_folder_dtw, exist_ok=True)

        dtw_results = calculate_dtw_cost_with_normalization(
            surrounding_values, surrounding_indices, integrated_areas_data,
            output_folder_dtw, plot_dtw_matrices_and_path)

        matrix_dtw, discontinuities_dtw, peaks_dtw = prepare_dtw_matrix(dtw_results)

        plot_dtw_heatmap(matrix_dtw, discontinuity_times, peaks_dtw, folder_path)

        dtw_df = pd.DataFrame(dtw_results).T
        dtw_df.index = discontinuity_times

        save_dtw_report(folder_path, dtw_df, discontinuity_info, metadata_df, discontinuity_times)

    return True


if __name__ == "__main__":
    for exp_config in experiment_configs:
        print(f"\n{'='*80}")
        print(f"PROCESSING EXPERIMENT: {exp_config['experiment']}")
        print(f"{'='*80}\n")

        try:
            success = process_experiment(
                exp_config, base_dir, analysis_configurations,
                potential_change_times, invert_pc1)

            if success:
                print(f"\nSuccessfully completed analysis for {exp_config['experiment']}")
            else:
                print(f"\nSkipped {exp_config['experiment']}")

        except Exception as e:
            print(f"\nERROR processing {exp_config['experiment']}: {str(e)}")
            traceback.print_exc()
            print(f"Continuing to next experiment...\n")
            continue

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS PROCESSED")
    print(f"{'='*80}\n")
