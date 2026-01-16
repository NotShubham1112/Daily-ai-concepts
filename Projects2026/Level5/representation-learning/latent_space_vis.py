import numpy as np
import matplotlib.pyplot as plt
from autoencoder import SimpleAutoencoder

def visualize_latent_space():
    # 1. Generate multi-cluster data in high dim
    N = 300
    centers = [np.array([2,2]), np.array([-2, -2]), np.array([2, -2])]
    data = []
    labels = []
    for i, c in enumerate(centers):
        cluster = c + np.random.randn(N//3, 2) * 0.5
        # Project to 10D
        proj = np.random.randn(2, 10)
        data.append(cluster @ proj)
        labels.extend([i] * (N//3))
    
    X = np.vstack(data)
    
    # 2. Train AE (conceptual / few steps)
    ae = SimpleAutoencoder(10, 2)
    # Simulation: In a real case, we'd run backprop here.
    # For vis, we'll just show the encoding.
    latent = ae.encode(X)
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title("Latent Space Visualization (AE Bottleneck)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.colorbar(label='Cluster ID')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    visualize_latent_space()
