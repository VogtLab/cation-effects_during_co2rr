import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_cluster_scatter(labels, output_dir):
    fig, ax = plt.subplots(figsize=(18, 8))
    scatter = ax.scatter(
        range(len(labels)),
        labels,
        c=labels,
        cmap='viridis',
        alpha=0.8
    )
    ax.set_xlabel('Time-series (PCs)', fontsize=18)
    ax.set_ylabel('Cluster Label', fontsize=18)
    ax.set_title('Clustering of PCs Across All Experiments', fontsize=18)
    plt.colorbar(scatter, ax=ax, label='Cluster Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_visualization.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'clustering_visualization.svg'))
    plt.close()


def plot_silhouette_scores(silhouette_scores, output_dir):
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(
        list(silhouette_scores.keys()),
        list(silhouette_scores.values()),
        marker='o',
        linestyle='-',
        linewidth=2
    )
    ax.set_xlabel('Number of Clusters', fontsize=18)
    ax.set_ylabel('Silhouette Score', fontsize=18)
    ax.set_title('Silhouette Scores for Different Numbers of Clusters', fontsize=18)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'silhouette_scores.svg'))
    plt.close()


def plot_centroids(centroids, output_dir, title='Cluster Centroids', filename='cluster_centroids'):
    fig, ax = plt.subplots(figsize=(18, 8))
    for cluster in centroids.index:
        ax.plot(
            centroids.columns,
            centroids.loc[cluster],
            label=f'Cluster {cluster}',
            marker='o',
            linewidth=2
        )
    ax.set_xlabel('Features (Reference Curves)', fontsize=18)
    ax.set_ylabel('Mean Alignment Costs', fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.legend(title='Clusters', fontsize=14, loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{filename}.svg'))
    plt.close()


def plot_cluster_heatmap(combined_data, output_dir, title='Normalized Heatmap with Cluster Separators',
                         filename='normalized_heatmap_by_cluster'):
    normalized_data = combined_data.copy()
    for column in combined_data.columns[:-1]:
        normalized_data[column] = (combined_data[column] - combined_data[column].min()) / (
            combined_data[column].max() - combined_data[column].min()
        )

    sorted_data = normalized_data.sort_values('Cluster')
    heatmap_data = sorted_data.drop(columns=['Cluster']).T

    cluster_sizes = sorted_data['Cluster'].value_counts(sort=False)
    cluster_boundaries = np.cumsum(cluster_sizes)

    fig, ax = plt.subplots(figsize=(100, 10))
    sns.heatmap(
        heatmap_data,
        cmap='Spectral_r',
        cbar_kws={'label': 'Normalized Values'},
        annot=False,
        linewidths=2,
        linecolor='gray'
    )

    for boundary in cluster_boundaries[:-1]:
        ax.axvline(boundary, color='black', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Experiments', fontsize=18)
    ax.set_ylabel('Features (Curves)', fontsize=18)
    ax.set_title(title, fontsize=18)

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(heatmap_data.index, fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{filename}.svg'))
    plt.close()


def plot_recluster_heatmap(subset, cluster_to_recluster, output_dir):
    heatmap_data = subset.sort_values('Recluster').drop(columns=['Recluster']).T
    cluster_sizes = subset['Recluster'].value_counts(sort=False)
    cluster_boundaries = np.cumsum(cluster_sizes)

    fig, ax = plt.subplots(figsize=(100, 10))
    sns.heatmap(
        heatmap_data,
        cmap='Spectral_r',
        cbar_kws={'label': 'Normalized Values'},
        annot=False,
        linewidths=2,
        linecolor='gray'
    )

    for boundary in cluster_boundaries[:-1]:
        ax.axvline(boundary, color='black', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Subset Experiments', fontsize=18)
    ax.set_ylabel('Features (Curves)', fontsize=18)
    ax.set_title(f'Recluster Heatmap for Cluster {cluster_to_recluster}', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recluster_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'recluster_heatmap.svg'))
    plt.close()
