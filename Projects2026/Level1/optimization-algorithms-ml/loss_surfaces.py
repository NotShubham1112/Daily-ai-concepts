import numpy as np

def convex_loss(w):
    return w[0]**2 + w[1]**2

def convex_grad(w):
    return np.array([2*w[0], 2*w[1]])

def non_convex_loss(w):
    return np.sin(w[0]) + 0.1 * w[0]**2 + w[1]**2

def non_convex_grad(w):
    return np.array([
        np.cos(w[0]) + 0.2*w[0],
        2*w[1]
    ])
