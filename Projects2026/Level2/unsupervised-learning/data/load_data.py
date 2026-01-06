import numpy as np
from .synthetic_blobs import generate

def load_synthetic():
    X, y = generate()
    return X.astype(np.float32), y
