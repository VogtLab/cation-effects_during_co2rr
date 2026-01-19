import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from alignment_references import normalize_to_magnitude_one


def minmax_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / range_val


def smooth_sequence(seq, window_length=50, polyorder=2):
    return savgol_filter(seq, window_length=window_length, polyorder=polyorder)


def dtw_distance_with_path_and_cost(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.zeros((n + 1, m + 1)) + np.inf
    dtw[0, 0] = 0
    traceback = np.zeros((n, m), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            options = [dtw[i - 1, j],
                       dtw[i, j - 1],
                       dtw[i - 1, j - 1]]
            tb_index = np.argmin(options)
            dtw[i, j] = cost + options[tb_index]
            traceback[i - 1, j - 1] = tb_index

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback[i, j]
        if tb_type == 0:
            i -= 1
        elif tb_type == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
        path.append((i, j))
    return dtw[n, m], path[::-1], dtw[1:, 1:]


def plot_dtw_matrices_and_path(dist_mat, cost_mat, path, x, y, output_folder, file_suffix):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplot(121)
    plt.title("Distance Matrix")
    plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    plt.subplot(122)
    plt.title("Cost Matrix")
    plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path)
    plt.plot(y_path, x_path)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"DTW_Matrices_{file_suffix}.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    for x_i, y_j in path:
        plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
    plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
    plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, f"DTW_Alignment_{file_suffix}.png"), dpi=300)
    plt.savefig(os.path.join(output_folder, f"DTW_Alignment_{file_suffix}.svg"), format="svg")
    plt.close()


def save_references_to_csv_and_plot(references, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    max_length = max(len(curve) for curve in references.values())
    padded_references = {name: np.pad(curve, (0, max_length - len(curve)), constant_values=np.nan)
                         for name, curve in references.items()}
    references_df = pd.DataFrame(padded_references)
    csv_path = os.path.join(output_dir, "reference_curves.csv")
    references_df.to_csv(csv_path, index=False)
    print(f"Reference curves saved to: {csv_path}")

    plt.figure(figsize=(12, 8))
    for name, curve in references.items():
        plt.plot(curve, label=name)

    plt.title("Reference Curves")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend(loc="best", fontsize='small')
    plt.grid()
    plot_path = os.path.join(output_dir, "reference_curves_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Reference curves plot saved to: {plot_path}")


def process_pca_file(file_path, references, output_dir, row_range=(80, 180), col_range=(1, 16)):
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(file_path, delimiter="\t", header=None).iloc[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    data_smoothed = data.apply(smooth_sequence, axis=0)
    data_normalized = minmax_scale(data_smoothed)

    results = []

    for col_idx in range(data_normalized.shape[1]):
        variable = data_normalized.iloc[:, col_idx].values

        for name, ref in references.items():
            ref_normalized = normalize_to_magnitude_one(ref)
            dist_mat = np.abs(np.subtract.outer(variable, ref_normalized))
            dtw_distance_value, path, cost_matrix = dtw_distance_with_path_and_cost(variable, ref_normalized)

            results.append({
                "File": file_path,
                "PC": col_idx + 1,
                "Reference": name,
                "DTW Distance": dtw_distance_value
            })

            plot_dtw_matrices_and_path(
                dist_mat, cost_matrix, path,
                variable, ref_normalized,
                output_dir,
                f"PC{col_idx + 1}_{name.replace(' ', '_')}"
            )

        best_match = min([r for r in results if r["PC"] == col_idx + 1], key=lambda x: x["DTW Distance"])
        for result in results:
            if result["PC"] == col_idx + 1:
                result["Best Match"] = "Yes" if result["Reference"] == best_match["Reference"] else "No"

    return results
