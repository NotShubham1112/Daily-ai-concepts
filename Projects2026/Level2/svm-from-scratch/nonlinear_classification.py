import numpy as np
import matplotlib.pyplot as plt
from svm_kernel import KernelSVM

def generate_xor(n=200):
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

X, y = generate_xor()

model = KernelSVM()
model.fit(X, y)
preds = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=preds, cmap="coolwarm")
plt.title("Non-linear Classification using Kernel SVM")
plt.show()
