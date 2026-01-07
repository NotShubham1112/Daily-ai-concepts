import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from activations import sigmoid, tanh, relu

def generate_data(n=200):
    X = np.random.randn(n, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)
    return X, y

X, y = generate_data()

losses = {}

for name in ["relu"]:
    model = NeuralNetwork([2, 16, 16, 1], lr=0.01)
    history = []

    for _ in range(200):
        pred = model.predict(X)
        history.append(np.mean((pred - y) ** 2))
        model.fit(X, y, epochs=1)

    losses[name] = history

for k, v in losses.items():
    plt.plot(v, label=k)

plt.title("Activation Function Training Behavior")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
