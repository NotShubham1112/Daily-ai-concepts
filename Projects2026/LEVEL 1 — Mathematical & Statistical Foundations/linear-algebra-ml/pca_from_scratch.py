import numpy as np

class PCAFromScratch:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray):
        # Step 1: Mean centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort by descending eigenvalues
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Step 5: Select top components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        return np.dot(X_reduced, self.components_.T) + self.mean_


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(200, 5)

    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    X_reduced = pca.transform(X)

    print("Reduced shape:", X_reduced.shape)
    print("Explained variance:", pca.explained_variance_)
