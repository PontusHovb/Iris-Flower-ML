import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.datasets import load_iris

K = 3
MAX_ITERS = 10000

# Self-implemented K-means algorithm
def kmeans(data, k, max_iters=100):
    n, dim = data.shape
    # Randomly initialize cluster centers
    centroids = data[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign data points to the nearest cluster center
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # Update cluster centers to the mean of each cluster
        new_centroids = np.array([np.mean(data[clusters == i], axis=0) \
                                   for i in range(k)])

        # If cluster centers no longer change, end the iteration
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, clusters

# Test the algorithms and output results
def run_algorithm(data, k, max_iters):
    centroids, kmeans_clusters = kmeans(data, k, max_iters)
    
    return centroids, kmeans_clusters

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Visualization
pca = PCA(n_components=2)
data = pca.fit_transform(data)

# Run the algorithms
centroids, kmeans_clusters = run_algorithm(data, K, MAX_ITERS)

# Print the results
print("K-means Centroids:", centroids)
print("K-means Clusters:", kmeans_clusters)

# Visualize the results
def visualize_results(data, k, centroids, kmeans_clusters):
    # Plot data points with colors representing clusters
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=kmeans_clusters, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()

# Visualize the results
visualize_results(data, K, centroids, kmeans_clusters)