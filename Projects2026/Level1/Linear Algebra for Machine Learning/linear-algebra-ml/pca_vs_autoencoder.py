import numpy as np
from pca_from_scratch import PCAFromScratch

class LinearAutoencoder:
    def __init__(self, input_dim, latent_dim, lr=0.01):
        self.W1 = np.random.randn(input_dim, latent_dim) * 0.01
        self.W2 = np.random.randn(latent_dim, input_dim) * 0.01
        self.lr = lr

    def forward(self, X):
        Z = X @ self.W1
        X_hat = Z @ self.W2
        return Z, X_hat

    def train(self, X, epochs=500):
        for _ in range(epochs):
            Z, X_hat = self.forward(X)
            error = X_hat - X

            # Gradients
            dW2 = Z.T @ error / len(X)
            dW1 = X.T @ (error @ self.W2.T) / len(X)

            # Update
            self.W1 -= self.lr * dW1
            self.W2 -= self.lr * dW2

    def encode(self, X):
        return X @ self.W1


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(500, 10)

    # PCA
    pca = PCAFromScratch(n_components=3)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Autoencoder
    ae = LinearAutoencoder(input_dim=10, latent_dim=3)
    ae.train(X)
    X_ae = ae.encode(X)

    print("PCA latent shape:", X_pca.shape)
    print("Autoencoder latent shape:", X_ae.shape)

    # Reconstruction error comparison
    X_pca_recon = pca.inverse_transform(X_pca)
    ae_recon = X_ae @ ae.W2

    print("PCA Reconstruction MSE:", np.mean((X - X_pca_recon) ** 2))
    print("Autoencoder Reconstruction MSE:", np.mean((X - ae_recon) ** 2))
