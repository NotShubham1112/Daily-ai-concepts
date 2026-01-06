import numpy as np

class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        cov = np.cov(X_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        components = eigvecs[:, :self.n_components]
        return X_centered @ components
