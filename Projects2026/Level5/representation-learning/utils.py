import numpy as np

def scale_data(X):
    return (X - X.min()) / (X.max() - X.min())
