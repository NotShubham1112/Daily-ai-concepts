import numpy as np

def random_crop(x, size):
    # Simplified augmentation
    start = np.random.randint(0, len(x) - size + 1)
    return x[start : start+size]
