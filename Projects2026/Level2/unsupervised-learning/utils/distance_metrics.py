import numpy as np

def euclidean(a, b):
    return np.linalg.norm(a - b)

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def cosine(a, b):
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
