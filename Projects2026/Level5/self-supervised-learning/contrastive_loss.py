import numpy as np

def info_nce_loss(features, temperature=0.1):
    """
    Implementation of InfoNCE (Contrastive) Loss.
    features: (batch_size, 2, embedding_dim) where 2 are positive pairs.
    """
    batch_size = features.shape[0]
    # Flatten to (2*batch_size, dim)
    obs = features.reshape(-1, features.shape[-1])
    
    # Compute similarity matrix
    # (2N, 2N)
    sim = (obs @ obs.T) / temperature
    
    # Mask out self-similarity
    mask = np.eye(2 * batch_size, dtype=bool)
    sim[mask] = -np.inf
    
    # For each i, the positive pair is at i+1 (if i is even) or i-1 (if i is odd)
    labels = np.zeros(2 * batch_size, dtype=int)
    for i in range(batch_size):
        labels[2*i] = 2*i + 1
        labels[2*i + 1] = 2*i
        
    # Log-softmax
    exp_sim = np.exp(sim)
    sum_exp = np.sum(exp_sim, axis=1)
    
    loss = -np.log(exp_sim[np.arange(2*batch_size), labels] / sum_exp)
    return np.mean(loss)

if __name__ == "__main__":
    # Simulate 4 positive pairs with dim 8
    feat = np.random.randn(4, 2, 8)
    # Make positive pairs similar
    feat[:, 1, :] = feat[:, 0, :] + np.random.normal(0, 0.01, (4, 8))
    
    loss = info_nce_loss(feat)
    print(f"Initial Contrastive Loss: {loss:.4f}")
