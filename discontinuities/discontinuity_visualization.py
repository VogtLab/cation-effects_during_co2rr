import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr


def plot_spearman_visualization(pca_values, integrated_values, discontinuity_time, output_folder):
    if len(pca_values) != len(integrated_values):
        min_len = min(len(pca_values), len(integrated_values))
        pca_values = pca_values[:min_len]
        integrated_values = integrated_values[:min_len]

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_values, integrated_values, color='blue')
    plt.title(f'PCA vs Integrated Areas (Discontinuity: {discontinuity_time:.2f} s)')
    plt.xlabel('PCA Values')
    plt.ylabel('Integrated Areas')
    plt.grid(True)

    pca_ranks = np.argsort(pca_values) + 1
    integrated_ranks = np.argsort(integrated_values) + 1

    for i, (pca, int_area) in enumerate(zip(pca_values, integrated_values)):
        plt.text(pca, int_area, f'({pca_ranks[i]},{integrated_ranks[i]})', fontsize=9, ha='right')

    plt.tight_layout()
    raw_plot_path = os.path.join(output_folder, f'spearman_raw_ranks_{discontinuity_time:.2f}.png')
    plt.savefig(raw_plot_path, dpi=300)
    plt.close()


def plot_spearman_visualization_with_arrows(pca_values, integrated_values, discontinuity_time, output_folder):
    if len(pca_values) != len(integrated_values):
        min_len = min(len(pca_values), len(integrated_values))
        pca_values = pca_values[:min_len]
        integrated_values = integrated_values[:min_len]

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_values, integrated_values, color='blue')
    plt.title(f'PCA vs Integrated Areas with Rank Differences (Discontinuity: {discontinuity_time:.2f} s)')
    plt.xlabel('PCA Values')
    plt.ylabel('Integrated Areas')
    plt.grid(True)

    pca_ranks = np.argsort(pca_values) + 1
    integrated_ranks = np.argsort(integrated_values) + 1

    for i, (pca, int_area) in enumerate(zip(pca_values, integrated_values)):
        plt.text(pca, int_area, f'({pca_ranks[i]},{integrated_ranks[i]})', fontsize=9, ha='right')

    for i in range(len(pca_values)):
        pca_rank = pca_ranks[i]
        int_rank = integrated_ranks[i]
        rank_diff = int_rank - pca_rank
        plt.arrow(pca_values[i], integrated_values[i], 0, rank_diff * 0.05,
                  head_width=0.1, head_length=0.05, fc='red', ec='red')
        plt.text(pca_values[i], integrated_values[i] + rank_diff * 0.025,
                 f'd={rank_diff}', fontsize=9, ha='center', color='green')

    spearman_corr, _ = spearmanr(pca_values, integrated_values)

    formula_text = r'$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$'
    plt.text(0.05, 0.95, f'{formula_text}\nSpearman Correlation: {spearman_corr:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    spearman_final_plot_path = os.path.join(output_folder, f'spearman_with_arrows_{discontinuity_time:.2f}.png')
    plt.savefig(spearman_final_plot_path, dpi=300)
    plt.close()


def explain_spearman_visualization(pca_values, integrated_values, discontinuity_time, output_folder):
    if len(pca_values) != len(integrated_values):
        min_len = min(len(pca_values), len(integrated_values))
        pca_values = pca_values[:min_len]
        integrated_values = integrated_values[:min_len]

    pca_ranks = np.argsort(pca_values) + 1
    integrated_ranks = np.argsort(integrated_values) + 1
    rank_diffs = integrated_ranks - pca_ranks
    rank_diffs_squared = rank_diffs ** 2
    sum_squared_diffs = np.sum(rank_diffs_squared)
    n = len(pca_values)

    spearman_corr = 1 - (6 * sum_squared_diffs) / (n * (n**2 - 1))

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_values, integrated_values, color='blue', s=100, edgecolor='black')
    plt.title(f'PCA vs Integrated Areas with Ranks (Discontinuity: {discontinuity_time:.2f} s)')
    plt.xlabel('PCA Values')
    plt.ylabel('Integrated Areas')
    plt.grid(True)

    for i, (pca, int_area) in enumerate(zip(pca_values, integrated_values)):
        plt.text(pca, int_area, f'({pca_ranks[i]},{integrated_ranks[i]})', fontsize=9, ha='right')

    for i in range(len(pca_values)):
        pca_rank = pca_ranks[i]
        int_rank = integrated_ranks[i]
        rank_diff = int_rank - pca_rank
        plt.arrow(pca_values[i], integrated_values[i], 0, rank_diff * 0.05,
                  head_width=0.1, head_length=0.05, fc='red', ec='red')
        plt.text(pca_values[i], integrated_values[i] + rank_diff * 0.025,
                 f'd={rank_diff}', fontsize=9, ha='center', color='green')

    formula_text = r'$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$'
    plt.text(0.05, 0.95, f'{formula_text}\nSpearman Correlation: {spearman_corr:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    spearman_final_plot_path = os.path.join(output_folder, f'spearman_explanation_{discontinuity_time:.2f}.png')
    spearman_final_plot_path_svg = os.path.join(output_folder, f'spearman_explanation_{discontinuity_time:.2f}.svg')
    plt.savefig(spearman_final_plot_path, dpi=300)
    plt.savefig(spearman_final_plot_path_svg, format='svg')
    plt.close()


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
    dist_cost_plot_path = os.path.join(output_folder, f"DTW_Matrices_{file_suffix}.png")
    plt.savefig(dist_cost_plot_path, dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    for x_i, y_j in path:
        plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
    plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
    plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
    plt.axis("off")

    alignment_plot_path_png = os.path.join(output_folder, f"DTW_Alignment_{file_suffix}.png")
    alignment_plot_path_svg = os.path.join(output_folder, f"DTW_Alignment_{file_suffix}.svg")
    plt.savefig(alignment_plot_path_png, dpi=300)
    plt.savefig(alignment_plot_path_svg, format='svg')
    plt.close()

    return dist_cost_plot_path, alignment_plot_path_png


def plot_discontinuities(time, pc1_smoothed, disc_times, disc_magnitudes,
                         potential_changes, output_path, filename):
    colors = ['#2c3e50', '#e74c3c']
    labels = ['PC1', 'Discontinuities']

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time, pc1_smoothed, label=labels[0], color=colors[0], linewidth=2)

    ax.scatter(disc_times, pc1_smoothed[np.searchsorted(time, disc_times)],
               color=colors[1], edgecolor='black', linewidth=1.5, s=100, label=labels[1])

    if potential_changes is not None:
        for pct_time in potential_changes:
            ax.axvline(x=pct_time, color='g', linestyle='--', alpha=0.5)

    ax.set_xlabel('Time (s)', fontsize=18)
    ax.set_ylabel('PC1', fontsize=18)
    ax.set_title('PC1 with Identified Discontinuities', fontsize=18)
    ax.legend(fontsize=18)
    ax.set_xlim(time[0], time[-1])
    ax.grid(False)

    message1 = f"Number of discontinuities: {len(disc_magnitudes)}"
    message2 = f"Average magnitude: {np.mean(np.abs(disc_magnitudes)):.2f}"
    message3 = f"Standard deviation of magnitude: {np.std(np.abs(disc_magnitudes)):.2f}"
    ax.text(0.05, 0.95, message1, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.text(0.05, 0.90, message2, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.text(0.05, 0.85, message3, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    plt.tight_layout()

    png_path = os.path.join(output_path, f"{filename}.png")
    svg_path = os.path.join(output_path, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.close()


def plot_spearman_heatmap(matrix, discontinuity_times, peaks, output_path, cmap='magma'):
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(matrix, cmap=cmap, annot=True, cbar=True,
                     xticklabels=peaks, yticklabels=discontinuity_times,
                     linewidths=0.5, annot_kws={"size": 10},
                     cbar_kws={"shrink": 0.8, "ticks": [-1, -0.5, 0, 0.5, 1]})

    ax.set_xlabel('FTIR Peaks', fontsize=14, labelpad=20)
    ax.set_ylabel('Discontinuities', fontsize=14, labelpad=10)
    ax.set_title('Spearman Correlation Coefficients', fontsize=18, pad=20)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    figure_filename_png = os.path.join(output_path, 'Spearman_Correlation_Matrix.png')
    figure_filename_svg = os.path.join(output_path, 'Spearman_Correlation_Matrix.svg')
    plt.savefig(figure_filename_png, dpi=300)
    plt.savefig(figure_filename_svg, format='svg')
    plt.close()


def plot_dtw_heatmap(matrix, discontinuity_times, peaks, output_path, cmap='magma'):
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(matrix, cmap=cmap, annot=True, cbar=True,
                     xticklabels=peaks, yticklabels=discontinuity_times,
                     linewidths=0.5, annot_kws={"size": 10},
                     cbar_kws={"shrink": 0.8, "ticks": [0, 50, 100, 150, 200]})

    ax.set_xlabel('FTIR Peaks', fontsize=14, labelpad=20)
    ax.set_ylabel('Discontinuities', fontsize=14, labelpad=10)
    ax.set_title('Non-Normalized DTW Alignment Costs', fontsize=18, pad=20)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'DTW_Correlation_Matrix_Non_Normalized.png'), dpi=300)
    plt.savefig(os.path.join(output_path, 'DTW_Correlation_Matrix_Non_Normalized.svg'), format='svg')
    plt.close()
