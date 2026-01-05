import numpy as np

def generate_surface(loss_fn, xlim=(-5,5), ylim=(-5,5), n=100):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[loss_fn(np.array([i, j])) for i in x] for j in y])
    return X, Y, Z
