import numpy as np
import matplotlib.pyplot as plt
from kernels import rbf_kernel

X = np.linspace(-3, 3, 100)

K = np.array([[rbf_kernel([x], [y]) for y in X] for x in X])

plt.imshow(K, cmap="viridis")
plt.title("RBF Kernel Similarity Matrix")
plt.colorbar()
plt.show()
