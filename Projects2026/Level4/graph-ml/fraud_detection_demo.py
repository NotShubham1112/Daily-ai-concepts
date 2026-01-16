import numpy as np

def detect_fraud_by_centrality(A, threshold=1.5):
    """
    Very simple demo: Fraudulent actors often have unusually high connectivity
    or specific structural patterns. We use degree centrality here.
    """
    degrees = np.sum(A, axis=1)
    avg_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    
    fraud_score = (degrees - avg_degree) / (std_degree + 1e-6)
    suspects = np.where(fraud_score > threshold)[0]
    
    return suspects, fraud_score

if __name__ == "__main__":
    # Random transaction graph
    N = 20
    adj = (np.random.rand(N, N) > 0.9).astype(float)
    # Add a "super-node" (potential fraudster)
    adj[5, :] = 1
    adj[:, 5] = 1
    
    suspects, scores = detect_fraud_by_centrality(adj)
    print(f"Suspected fraudulent nodes: {suspects}")
    print(f"Score for node 5: {scores[5]:.2f}")
