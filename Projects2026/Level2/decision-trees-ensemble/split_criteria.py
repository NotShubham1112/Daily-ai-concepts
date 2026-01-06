import numpy as np

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def information_gain(y, y_left, y_right, criterion="entropy"):
    if criterion == "entropy":
        parent = entropy(y)
        left = entropy(y_left)
        right = entropy(y_right)
    else:
        parent = gini(y)
        left = gini(y_left)
        right = gini(y_right)

    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)

    return parent - (w_left * left + w_right * right)
