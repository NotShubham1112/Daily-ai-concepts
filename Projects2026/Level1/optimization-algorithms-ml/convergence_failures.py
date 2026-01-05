import numpy as np
import matplotlib.pyplot as plt
from optimizers import GradientDescent
from loss_surfaces import non_convex_loss, non_convex_grad

opt = GradientDescent(lr=0.5)  # intentionally high LR
w = np.array([2.5, 2.5])
path = [w.copy()]

for _ in range(50):
    grad = non_convex_grad(w)
    w = opt.step(w, grad)
    path.append(w.copy())

path = np.array(path)

plt.plot(path[:, 0], label="w1")
plt.plot(path[:, 1], label="w2")
plt.title("Divergence on Non-Convex Loss (High Learning Rate)")
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.grid()
plt.show()
