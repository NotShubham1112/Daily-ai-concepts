import numpy as np

def entropy(p, eps=1e-10):
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log(p))


def expectation(x, p):
    return np.sum(x * p)


def variance(x, p):
    mu = expectation(x, p)
    return np.sum((x - mu) ** 2 * p)
