import numpy as np
import matplotlib.pyplot as plt
from optimizers import GradientDescent, Momentum, RMSProp, Adam
from loss_surfaces import convex_loss, convex_grad

optimizers = {
    "GD": GradientDescent(0.1),
    "Momentum": Momentum(0.1),
    "RMSProp": RMSProp(0.1),
    "Adam": Adam(0.1)
}

def run(optimizer, steps=50):
    w = np.array([5.0, 5.0])
    path = [w.copy()]
    for _ in range(steps):
        grad = convex_grad(w)
        w = optimizer.step(w, grad)
        path.append(w.copy())
    return np.array(path)

plt.figure(figsize=(8, 6))

for name, opt in optimizers.items():
    path = run(opt)
    plt.plot(path[:, 0], path[:, 1], marker='o', label=name)

plt.title("Optimizer Paths on Convex Loss Surface")
plt.xlabel("w1")
plt.ylabel("w2")
plt.legend()
plt.grid()
plt.show()
