import pandas as pd
import os
import matplotlib as mpl

from clustering import (
    prepare_combined_data,
    normalize_and_cluster,
    find_optimal_clusters,
    calculate_cluster_metrics,
    calculate_centroids,
    calculate_cosine_similarity,
    calculate_regression_scores,
    recluster_subset
)
from clustering_visualization import (
    plot_cluster_scatter,
    plot_silhouette_scores,
    plot_centroids,
    plot_cluster_heatmap,
    plot_recluster_heatmap
)

mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'


input_file = './data/alignment_costs.csv'
output_dir = './clustering_output'

cluster_to_recluster = 2


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data = pd.read_csv(input_file)

    print("Preparing combined data...")
    combined_data = prepare_combined_data(data)

    print("Performing initial clustering...")
    combined_data, normalized_data, similarity_matrix, labels = normalize_and_cluster(combined_data, n_clusters=2)

    output_file = os.path.join(output_dir, 'combined_clustering_results.csv')
    combined_data.to_csv(output_file)
    print(f"Initial clustering results saved to {output_file}")

    plot_cluster_scatter(labels, output_dir)

    print("\nFinding optimal number of clusters...")
    optimal_clusters, silhouette_scores = find_optimal_clusters(normalized_data, similarity_matrix)

    plot_silhouette_scores(silhouette_scores, output_dir)

    print(f"\nReclustering with optimal clusters ({optimal_clusters})...")
    combined_data, normalized_data, similarity_matrix, labels = normalize_and_cluster(
        combined_data.drop(columns=['Cluster']), n_clusters=optimal_clusters
    )

    output_file = os.path.join(output_dir, 'combined_clustering_results_optimal.csv')
    combined_data.to_csv(output_file)
    print(f"Optimal clustering results saved to {output_file}")

    db_index = calculate_cluster_metrics(normalized_data, labels)

    print("\nCalculating centroids...")
    centroids = calculate_centroids(combined_data)
    centroids_file = os.path.join(output_dir, 'cluster_centroids.csv')
    centroids.to_csv(centroids_file)
    print(f"Cluster centroids saved to {centroids_file}")

    plot_centroids(centroids, output_dir)

    print("\nCalculating cosine similarity...")
    similarity_df = calculate_cosine_similarity(normalized_data, centroids, combined_data)
    similarity_file = os.path.join(output_dir, 'time_series_cluster_similarity.csv')
    similarity_df.to_csv(similarity_file)
    print(f"Cosine similarity saved to {similarity_file}")

    print("\nCalculating regression scores...")
    regression_df = calculate_regression_scores(normalized_data, centroids.drop(columns=['Cluster'], errors='ignore'), combined_data)
    regression_file = os.path.join(output_dir, 'time_series_cluster_regression.csv')
    regression_df.to_csv(regression_file)
    print(f"Regression scores saved to {regression_file}")

    print("\nGenerating heatmap...")
    plot_cluster_heatmap(combined_data, output_dir)

    if cluster_to_recluster is not None:
        print(f"\nReclustering cluster {cluster_to_recluster}...")
        recluster_dir = os.path.join(output_dir, f"recluster_cluster_{cluster_to_recluster}")
        os.makedirs(recluster_dir, exist_ok=True)

        subset, normalized_subset, recluster_similarity, recluster_labels = recluster_subset(
            combined_data, cluster_to_recluster, n_reclusters=2
        )

        if subset is not None:
            recluster_file = os.path.join(recluster_dir, 'recluster_results.csv')
            subset.to_csv(recluster_file)
            print(f"Recluster results saved to {recluster_file}")

            recluster_centroids = subset.groupby('Recluster').mean()
            recluster_centroids_file = os.path.join(recluster_dir, 'recluster_centroids.csv')
            recluster_centroids.to_csv(recluster_centroids_file)
            print(f"Recluster centroids saved to {recluster_centroids_file}")

            plot_centroids(recluster_centroids, recluster_dir,
                           title=f'Recluster Centroids for Cluster {cluster_to_recluster}',
                           filename='recluster_centroids')

            plot_recluster_heatmap(subset, cluster_to_recluster, recluster_dir)

    print("\nClustering analysis complete!")
