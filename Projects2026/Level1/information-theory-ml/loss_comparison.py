import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred, eps=1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

x = np.linspace(-5, 5, 400)
y_true = (x > 0).astype(int)
y_pred = 1 / (1 + np.exp(-x))

mse_loss = [mse(y_true, y_pred)]
ce_loss = [cross_entropy(y_true, y_pred)]

plt.bar(["MSE", "Cross-Entropy"], [mse_loss[0], ce_loss[0]])
plt.title("Loss Comparison for Binary Classification")
plt.ylabel("Loss Value")
plt.grid()
plt.show()
