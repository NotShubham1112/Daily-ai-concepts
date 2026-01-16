import numpy as np

class BayesianLinearRegression:
    def __init__(self, alpha=1e-3, beta=1.0):
        """
        alpha: precision of the prior (lambda/sigma^2)
        beta: precision of the noise
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        
    def fit(self, X, y):
        """
        Closed form solution for Bayesian Linear Regression:
        p(w|t) = N(w | m_N, S_N)
        S_N^-1 = alpha * I + beta * Phi.T @ Phi
        m_N = beta * S_N @ Phi.T @ y
        """
        d = X.shape[1]
        I = np.eye(d)
        
        # Calculate covariance S_N
        self.cov = np.linalg.inv(self.alpha * I + self.beta * (X.T @ X))
        
        # Calculate mean m_N
        self.mean = self.beta * (self.cov @ X.T @ y)
        
    def predict(self, X):
        """
        Predictive distribution: p(y|x, X, y) = N(y | m_N.T @ x, sigma^2(x))
        sigma^2(x) = 1/beta + x.T @ S_N @ x
        """
        mu = X @ self.mean
        variance = 1.0 / self.beta + np.sum(X @ self.cov * X, axis=1)
        return mu, np.sqrt(variance)
