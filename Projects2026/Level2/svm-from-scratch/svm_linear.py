import numpy as np

class LinearSVM:
    def __init__(self, lr=0.001, lambda_=0.01, epochs=1000):
        self.lr = lr
        self.lambda_ = lambda_
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        y = np.where(y == 0, -1, 1)  # convert to {-1, +1}
        self.w = np.zeros(d)

        for _ in range(self.epochs):
            for i in range(n):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_ * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_ * self.w - y[i] * X[i])
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(X @ self.w + self.b)
