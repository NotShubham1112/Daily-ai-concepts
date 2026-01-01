"""
Bias-Variance Simulation
------------------------
Simple polynomial fitting intuition.
"""

import numpy as np

np.random.seed(0)

X = np.linspace(0, 1, 10)
true_y = 2 * X
noise = np.random.normal(0, 0.2, size=len(X))
y = true_y + noise

# Underfitting (constant model)
underfit = np.mean(y)

# Overfitting (high-degree polynomial)
coeffs = np.polyfit(X, y, deg=9)

print("Underfitting prediction:", underfit)
print("Overfitting coefficients:", coeffs)
