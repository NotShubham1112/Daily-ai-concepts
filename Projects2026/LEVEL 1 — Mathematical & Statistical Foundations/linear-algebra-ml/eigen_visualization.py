import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate correlated data
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean, cov, 300)

# Mean center
X_centered = X - np.mean(X, axis=0)

# Covariance and eigen decomposition
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Plot
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.3)

origin = np.mean(X_centered, axis=0)
for i in range(2):
    vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 3
    plt.arrow(
        origin[0], origin[1],
        vec[0], vec[1],
        color='red',
        width=0.05,
        head_width=0.2
    )

plt.title("Eigenvectors Represent Principal Variance Directions")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid()
plt.show()
