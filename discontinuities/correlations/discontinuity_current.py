import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr


def load_current_file(current_directory, experiment_file):
    for file_name in os.listdir(current_directory):
        if file_name.endswith(experiment_file):
            return os.path.join(current_directory, file_name)

    raise FileNotFoundError(f"Current file {experiment_file} not found in {current_directory}")


def load_current_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    time_column = data.iloc[:, 0].values
    current_data = data.drop(columns=data.columns[0])
    return time_column, current_data


def calculate_spearman_for_discontinuities_time_based(surrounding_times, surrounding_values,
                                                       integrated_time, integrated_areas_data):
    spearman_coefficients = {}

    for i, times in enumerate(surrounding_times):
        spearman_coefficients[f'Discontinuity_{i+1}'] = {}

        closest_indices = [
            np.where(np.abs(integrated_time - t) == np.min(np.abs(integrated_time - t)))[0][0]
            for t in times
        ]

        for peak in integrated_areas_data.columns:
            integrated_values = integrated_areas_data[peak].values[closest_indices]

            pca_values = surrounding_values[i]
            if len(pca_values) == len(integrated_values):
                spearman_corr, _ = spearmanr(pca_values, integrated_values)
                spearman_coefficients[f'Discontinuity_{i+1}'][peak] = spearman_corr

    return spearman_coefficients


def calculate_dtw_cost_time_based(surrounding_times, surrounding_values, integrated_time,
                                   integrated_areas_data, output_folder, dp_func, plot_func=None):
    dtw_costs = {}

    for i, times in enumerate(surrounding_times):
        dtw_costs[f'Discontinuity_{i+1}'] = {}

        closest_indices = [
            np.where(np.abs(integrated_time - t) == np.min(np.abs(integrated_time - t)))[0][0]
            for t in times
        ]

        for peak in integrated_areas_data.columns:
            integrated_values = integrated_areas_data[peak].values[closest_indices]
            pca_values = surrounding_values[i]

            pca_values_normalized = normalize_sequence(pca_values)
            integrated_values_normalized = normalize_sequence(integrated_values)

            if len(pca_values_normalized) == len(integrated_values_normalized):
                dist_mat = np.abs(np.subtract.outer(pca_values_normalized, integrated_values_normalized))

                path, cost_mat = dp_func(dist_mat)

                dtw_cost = cost_mat[-1, -1]

                dtw_costs[f'Discontinuity_{i+1}'][peak] = dtw_cost

                if plot_func is not None:
                    file_suffix = f"Disc_{i+1}_Peak_{peak.replace(' ', '_')}"
                    plot_func(dist_mat, cost_mat, path, pca_values_normalized,
                              integrated_values_normalized, output_folder, file_suffix)

    return dtw_costs


def normalize_sequence(seq):
    min_val = np.min(seq)
    max_val = np.max(seq)
    return (seq - min_val) / (max_val - min_val) if max_val - min_val != 0 else seq
