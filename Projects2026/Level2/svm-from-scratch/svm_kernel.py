import numpy as np
from kernels import rbf_kernel

class KernelSVM:
    def __init__(self, kernel=rbf_kernel, lr=0.001, epochs=500):
        self.kernel = kernel
        self.lr = lr
        self.epochs = epochs
        self.alpha = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        n = X.shape[0]
        y = np.where(y == 0, -1, 1)
        self.alpha = np.zeros(n)
        self.X = X
        self.y = y

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])

        for _ in range(self.epochs):
            for i in range(n):
                margin = np.sum(self.alpha * self.y * K[:, i])
                if y[i] * margin < 1:
                    self.alpha[i] += self.lr

    def project(self, X):
        result = []
        for x in X:
            s = 0
            for alpha, y_i, x_i in zip(self.alpha, self.y, self.X):
                s += alpha * y_i * self.kernel(x_i, x)
            result.append(s)
        return np.array(result)

    def predict(self, X):
        return np.sign(self.project(X))
