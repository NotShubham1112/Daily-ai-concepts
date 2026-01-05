import numpy as np

def normalize(p):
    p = np.array(p, dtype=float)
    return p / np.sum(p)
