import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        self.X = X
        n_samples, n_features = X.shape

        # Random centroid initialization
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(n_samples, self.k, replace=False)]

        for _ in range(self.max_iters):
            labels = self._assign_clusters()
            new_centroids = self._update_centroids(labels)

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return labels

    def _assign_clusters(self):
        distances = np.linalg.norm(
            self.X[:, np.newaxis] - self.centroids, axis=2
        )
        return np.argmin(distances, axis=1)

    def _update_centroids(self, labels):
        return np.array([
            self.X[labels == i].mean(axis=0)
            for i in range(self.k)
        ])
