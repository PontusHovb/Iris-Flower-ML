import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.datasets import load_iris

K = 3
MAX_ITERS = 10000

# Self-implemented Gaussian Mixture Model EM algorithm
def gaussian_mixture_em(data, k, max_iters=100):
    n, dim = data.shape

    # Initialize Gaussian Mixture Model parameters
    means = data[np.random.choice(n, k, replace=False)]
    covariances = [np.eye(dim) for _ in range(k)]

    weights = np.ones(k) / k
    
    for _ in range(max_iters):
        # E-step: Calculate posterior probabilities
        posteriors = np.array([weights[i] * \
                    multivariate_normal.pdf(data, means[i], covariances[i]) \
                    for i in range(k)]).T
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        
        # M-step: Update model parameters
        Nk = posteriors.sum(axis=0)
        weights = Nk / n
        means = np.dot(posteriors.T, data) / Nk[:, np.newaxis]
        covariances = [np.dot((data - means[i]).T, \
                    (posteriors[:, i] * (data - means[i]).T).T) / Nk[i] \
                    for i in range(k)]
        
        # Calculate log-likelihood and check for convergence
        log_likelihood = np.sum(np.log(np.sum([weights[i] * \
                    multivariate_normal.pdf(data, means[i], covariances[i]) \
                    for i in range(k)], axis=0)))
        if np.isnan(log_likelihood):
            break
    clusters = np.argmax(posteriors, axis=1)

    return means, covariances, weights, clusters


# Test the algorithms and output results
def run_algorithm(data, k, max_iters):
    means, covariances, weights, gmm_clusters = gaussian_mixture_em(data, k, max_iters)
    
    return gmm_clusters, means, covariances, weights

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Visualization
pca = PCA(n_components=2)
data = pca.fit_transform(data)

# Run the algorithms
gmm_clusters, means, covariances, weights = run_algorithm(data, K, MAX_ITERS)

# Print the results
print("Gaussian Mixture Means:", means)
print("Gaussian Mixture Covariances:", covariances)
print("Gaussian Mixture Weights:", weights)
print("Gaussian Mixture Clusters:", gmm_clusters)

# Visualize the results
def visualize_results(data, k, gmm_clusters, means, covariances):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=gmm_clusters, cmap='viridis', alpha=0.5)
    for i in range(k):
        x, y = np.mgrid[min(data[:, 0]):max(data[:, 0]):100j, min(data[:, 1]):max(data[:, 1]):100j]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        plt.contour(x, y, rv.pdf(pos), colors='k', alpha=0.5)
    plt.title('Gaussian Mixture Model')
    plt.show()

# Visualize the results
visualize_results(data, K, gmm_clusters, means, covariances)