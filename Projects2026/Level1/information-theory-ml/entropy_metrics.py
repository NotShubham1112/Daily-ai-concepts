import numpy as np

def entropy(p, eps=1e-10):
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log2(p))


def cross_entropy(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return -np.sum(p * np.log2(q))


def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log2(p / q))


if __name__ == "__main__":
    p = np.array([0.4, 0.6])
    q = np.array([0.5, 0.5])

    print("Entropy:", entropy(p))
    print("Cross-Entropy:", cross_entropy(p, q))
    print("KL Divergence:", kl_divergence(p, q))
