import numpy as np
from scipy.stats import spearmanr


def calculate_spearman_for_discontinuities(surrounding_values, surrounding_indices, integrated_areas_data):
    spearman_coefficients = {}

    for i, indices in enumerate(surrounding_indices):
        spearman_coefficients[f'Discontinuity_{i+1}'] = {}

        adjusted_indices = indices + 1

        for peak in integrated_areas_data.columns:
            integrated_values = integrated_areas_data[peak].values[adjusted_indices]

            pca_values = surrounding_values[i]
            if len(pca_values) == len(integrated_values):
                spearman_corr, _ = spearmanr(pca_values, integrated_values)
                spearman_coefficients[f'Discontinuity_{i+1}'][peak] = spearman_corr

    return spearman_coefficients


def prepare_spearman_matrix(spearman_results):
    discontinuities = list(spearman_results.keys())
    peaks = list(next(iter(spearman_results.values())).keys())

    matrix = np.zeros((len(discontinuities), len(peaks)))

    for i, disc in enumerate(discontinuities):
        for j, peak in enumerate(peaks):
            matrix[i, j] = spearman_results[disc][peak]

    return matrix, discontinuities, peaks


def dp(dist_mat):
    N, M = dist_mat.shape
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],
                cost_mat[i, j + 1],
                cost_mat[i + 1, j]]
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            i -= 1
            j -= 1
        elif tb_type == 1:
            i -= 1
        elif tb_type == 2:
            j -= 1
        path.append((i, j))
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)


def normalize_sequence(seq):
    min_val = np.min(seq)
    max_val = np.max(seq)
    return (seq - min_val) / (max_val - min_val) if max_val - min_val != 0 else seq


def calculate_dtw_cost_with_normalization(surrounding_values, surrounding_indices, integrated_areas_data,
                                           output_folder, plot_func=None):
    dtw_costs = {}

    for i, indices in enumerate(surrounding_indices):
        dtw_costs[f'Discontinuity_{i+1}'] = {}

        adjusted_indices = indices + 1

        for peak in integrated_areas_data.columns:
            integrated_values = integrated_areas_data[peak].values[adjusted_indices]
            pca_values = surrounding_values[i]

            pca_values_normalized = normalize_sequence(pca_values)
            integrated_values_normalized = normalize_sequence(integrated_values)

            if len(pca_values_normalized) == len(integrated_values_normalized):
                dist_mat = np.abs(np.subtract.outer(pca_values_normalized, integrated_values_normalized))

                path, cost_mat = dp(dist_mat)

                dtw_cost = cost_mat[-1, -1]

                dtw_costs[f'Discontinuity_{i+1}'][peak] = dtw_cost

                if plot_func is not None:
                    file_suffix = f"Disc_{i+1}_Peak_{peak.replace(' ', '_')}"
                    plot_func(dist_mat, cost_mat, path, pca_values_normalized,
                              integrated_values_normalized, output_folder, file_suffix)

    return dtw_costs


def prepare_dtw_matrix(dtw_results):
    discontinuities = list(dtw_results.keys())
    peaks = list(next(iter(dtw_results.values())).keys())

    matrix = np.zeros((len(discontinuities), len(peaks)))

    for i, disc in enumerate(discontinuities):
        for j, peak in enumerate(peaks):
            matrix[i, j] = dtw_results[disc][peak]

    return matrix, discontinuities, peaks
