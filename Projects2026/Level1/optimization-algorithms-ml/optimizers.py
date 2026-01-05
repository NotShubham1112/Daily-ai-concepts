import numpy as np

class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads


class Momentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(grads)
        self.v = self.beta * self.v + grads
        return params - self.lr * self.v


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = None

    def step(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(grads)
        self.cache = self.beta * self.cache + (1 - self.beta) * grads**2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
