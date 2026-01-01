"""
Matrix Transformation Experiment
--------------------------------
Shows how matrices transform vectors.
"""

import numpy as np

# Transformation matrix
W = np.array([
    [2, 0],
    [0, 1]
])

# Input vector
x = np.array([3, 4])

y = W @ x

print("Input vector:", x)
print("Transformed vector:", y)
