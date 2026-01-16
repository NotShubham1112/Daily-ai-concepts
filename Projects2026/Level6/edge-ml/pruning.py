import numpy as np

def magnitude_pruning(weights, sparsity_ratio=0.5):
    """
    Sets weights with smallest absolute values to zero.
    """
    abs_weights = np.abs(weights)
    threshold = np.percentile(abs_weights, sparsity_ratio * 100)
    
    pruned_weights = weights.copy()
    pruned_weights[abs_weights < threshold] = 0
    return pruned_weights

if __name__ == "__main__":
    w = np.random.randn(10, 10)
    w_pruned = magnitude_pruning(w, 0.7)
    
    nz = np.count_nonzero(w_pruned)
    print(f"Original weights: {w.size}")
    print(f"Non-zero weights after 70% pruning: {nz}")
