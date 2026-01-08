import numpy as np
import matplotlib.pyplot as plt

T = 50
Whh = np.array([[0.5]])
grad = 1.0
grads = []

for _ in range(T):
    grad *= Whh[0, 0]
    grads.append(grad)

plt.plot(grads)
plt.title("Vanishing Gradient in RNN")
plt.xlabel("Time Step")
plt.ylabel("Gradient Magnitude")
plt.show()
