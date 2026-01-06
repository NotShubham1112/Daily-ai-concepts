import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, lambda_=0.0):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)

            dw = (X.T @ (y_hat - y)) / n + self.lambda_ * self.w
            db = np.mean(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
