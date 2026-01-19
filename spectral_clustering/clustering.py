import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LinearRegression


def prepare_combined_data(data):
    experiments = data.columns[2:]
    combined_data = []

    for experiment in experiments:
        alignment_data = data[[data.columns[0], data.columns[1], experiment]]
        pivot_data = alignment_data.pivot(index='PC', columns='Reference', values=experiment)
        pivot_data = pivot_data.fillna(0)
        pivot_data.index = [f"{pc}_{experiment}" for pc in pivot_data.index]
        combined_data.append(pivot_data)

    combined_data = pd.concat(combined_data, axis=0)
    combined_data = combined_data.apply(pd.to_numeric, errors='coerce')
    combined_data = combined_data.dropna()
    combined_data.columns = combined_data.columns.astype(str)

    return combined_data


def normalize_and_cluster(combined_data, n_clusters=2):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(combined_data)

    similarity_matrix = rbf_kernel(normalized_data)

    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(similarity_matrix)

    combined_data['Cluster'] = labels

    return combined_data, normalized_data, similarity_matrix, labels


def find_optimal_clusters(normalized_data, similarity_matrix, cluster_range=(2, 11)):
    silhouette_scores = {}

    for n_clusters in range(cluster_range[0], cluster_range[1]):
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
        labels = clustering.fit_predict(similarity_matrix)
        score = silhouette_score(normalized_data, labels, metric='euclidean')
        silhouette_scores[n_clusters] = score
        print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

    optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Optimal Number of Clusters: {optimal_clusters}")

    return optimal_clusters, silhouette_scores


def calculate_cluster_metrics(normalized_data, labels):
    db_index = davies_bouldin_score(normalized_data, labels)
    print(f"Davies-Bouldin Index: {db_index}")
    return db_index


def calculate_centroids(combined_data):
    centroids = combined_data.groupby('Cluster').mean()
    return centroids


def calculate_cosine_similarity(normalized_data, centroids, combined_data):
    if 'Cluster' in centroids.columns:
        centroids = centroids.drop(columns=['Cluster'])

    similarity_scores = cosine_similarity(normalized_data, centroids)
    similarity_df = pd.DataFrame(
        similarity_scores,
        columns=[f"Cluster_{int(cluster)}" for cluster in centroids.index],
        index=combined_data.index
    )
    similarity_df['Assigned_Cluster'] = combined_data['Cluster']

    return similarity_df


def calculate_regression_scores(normalized_data, centroids, combined_data):
    regression_scores = []

    for i, row in enumerate(normalized_data):
        scores = []
        for centroid in centroids.values:
            model = LinearRegression()
            model.fit(centroid.reshape(-1, 1), row)
            r2_score = model.score(centroid.reshape(-1, 1), row)
            scores.append(r2_score)
        regression_scores.append(scores)

    regression_df = pd.DataFrame(
        regression_scores,
        columns=[f"Cluster_{int(cluster)}" for cluster in centroids.index],
        index=combined_data.index
    )
    regression_df['Assigned_Cluster'] = combined_data['Cluster']

    return regression_df


def recluster_subset(data, cluster_to_recluster, n_reclusters=2):
    if cluster_to_recluster not in data['Cluster'].unique():
        print(f"Cluster {cluster_to_recluster} does not exist. Skipping reclustering.")
        return None, None, None, None

    subset = data[data['Cluster'] == cluster_to_recluster].drop(columns=['Cluster'])
    subset.columns = subset.columns.astype(str)

    scaler = StandardScaler()
    normalized_subset = scaler.fit_transform(subset)
    similarity_matrix = rbf_kernel(normalized_subset)

    reclustering = SpectralClustering(n_clusters=n_reclusters, affinity='precomputed', random_state=0)
    recluster_labels = reclustering.fit_predict(similarity_matrix)
    subset['Recluster'] = recluster_labels

    return subset, normalized_subset, similarity_matrix, recluster_labels
