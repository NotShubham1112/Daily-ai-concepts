import numpy as np

def create_random_graph(n, p):
    return (np.random.rand(n, n) < p).astype(float)
