import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)
