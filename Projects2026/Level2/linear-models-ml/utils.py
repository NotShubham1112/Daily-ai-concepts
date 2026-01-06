import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
