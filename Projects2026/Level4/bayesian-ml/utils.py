import numpy as np

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    return (X - mean) / std
