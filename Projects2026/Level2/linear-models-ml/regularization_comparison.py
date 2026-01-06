import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

np.random.seed(0)

X = np.random.randn(100, 10)
true_w = np.array([5, 0, 0, 3, 0, 0, 2, 0, 0, 0])
y = X @ true_w + np.random.randn(100)

models = {
    "No Reg": LinearRegression(),
    "L1": LinearRegression(reg='l1', lambda_=0.1),
    "L2": LinearRegression(reg='l2', lambda_=0.1)
}

weights = {}

for name, model in models.items():
    model.fit(X, y)
    weights[name] = model.w

plt.figure(figsize=(10, 5))
for name, w in weights.items():
    plt.plot(w, marker='o', label=name)

plt.title("L1 vs L2 Regularization Effect on Weights")
plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.legend()
plt.grid()
plt.show()
