import numpy as np

def normalize_adjacency(A):
    """
    Computes \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}
    """
    A_hat = A + np.eye(A.shape[0])
    D_hat = np.diag(np.sum(A_hat, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D_hat))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

class GCNLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        
    def forward(self, X, A_norm):
        """
        X: Node features (N, in_features)
        A_norm: Normalized adjacency matrix (N, N)
        """
        # Message passing: A_norm @ X aggregated neighbors
        # Transformation: @ W
        return A_norm @ X @ self.W

def relu(x):
    return np.maximum(0, x)
