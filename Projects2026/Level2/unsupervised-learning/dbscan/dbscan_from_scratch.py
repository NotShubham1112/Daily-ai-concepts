import numpy as np

class DBSCANScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.X = X
        self.labels = np.full(len(X), -1)
        cluster_id = 0

        for i in range(len(X)):
            if self.labels[i] != -1:
                continue

            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self._expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels

    def _region_query(self, idx):
        distances = np.linalg.norm(self.X - self.X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, idx, neighbors, cluster_id):
        self.labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            point = neighbors[i]
            if self.labels[point] == -1:
                self.labels[point] = cluster_id
                new_neighbors = self._region_query(point)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            i += 1
