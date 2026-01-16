import numpy as np

def apply_masking(x, mask_ratio=0.15):
    """
    Randomly masks percentage of the input vector.
    """
    n = len(x)
    mask = np.random.rand(n) < mask_ratio
    x_masked = x.copy()
    x_masked[mask] = 0.0 # Standard mask value
    return x_masked, mask

def masking_task_demo():
    # Sequence of length 10
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    x_masked, mask = apply_masking(x)
    
    print(f"Original: {x}")
    print(f"Masked:   {x_masked}")
    print(f"Indices to predict: {np.where(mask)[0]}")

if __name__ == "__main__":
    masking_task_demo()
