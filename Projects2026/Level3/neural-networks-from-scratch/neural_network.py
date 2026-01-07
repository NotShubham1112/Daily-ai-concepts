import numpy as np
from activations import relu, relu_grad
from losses import mse, mse_grad

class NeuralNetwork:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i+1])))

    def forward(self, X):
        activations = [X]
        zs = []

        for W, b in zip(self.weights, self.biases):
            z = activations[-1] @ W + b
            zs.append(z)
            a = relu(z)
            activations.append(a)

        return activations, zs

    def backward(self, X, y, activations, zs):
        grads_w = []
        grads_b = []

        y_pred = activations[-1]
        delta = mse_grad(y, y_pred) * relu_grad(zs[-1])

        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, activations[i].T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))

            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_grad(zs[i-1])

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            activations, zs = self.forward(X)
            self.backward(X, y, activations, zs)

    def predict(self, X):
        return self.forward(X)[0][-1]
