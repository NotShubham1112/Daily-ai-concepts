import numpy as np

def explained_variance_ratio(eigenvalues):
    return eigenvalues / np.sum(eigenvalues)
