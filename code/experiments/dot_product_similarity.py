"""
Dot Product Similarity
----------------------
Measures similarity between two vectors.
"""

import numpy as np

# Example vectors (embeddings-like)
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([-1, -2, -3])

print("v1 · v2 (similar):", np.dot(v1, v2))
print("v1 · v3 (opposite):", np.dot(v1, v3))
