import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000, reg=None, lambda_=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg          # None, 'l1', 'l2'
        self.lambda_ = lambda_
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (X.T @ error) / n
            db = np.mean(error)

            if self.reg == 'l2':
                dw += self.lambda_ * self.w
            elif self.reg == 'l1':
                dw += self.lambda_ * np.sign(self.w)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b
