import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

X = np.random.randn(200, 10)
y = np.random.randn(200, 1)

model = NeuralNetwork([10, 20, 20, 1], lr=0.001)

gradient_norms = []

for _ in range(200):
    activations, zs = model.forward(X)
    model.backward(X, y, activations, zs)

    norms = [np.linalg.norm(w) for w in model.weights]
    gradient_norms.append(norms)

gradient_norms = np.array(gradient_norms)

for i in range(gradient_norms.shape[1]):
    plt.plot(gradient_norms[:, i], label=f"Layer {i+1}")

plt.title("Gradient Flow Across Layers")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.legend()
plt.show()
