import numpy as np

def get_saliency_map(input_data, gradients):
    """
    Saliency map: gradient of the output w.r.t the input.
    """
    # Simple absolute value of gradients
    saliency = np.abs(gradients)
    # Normalize for visualization
    denom = np.max(saliency) + 1e-10
    return saliency / denom

if __name__ == "__main__":
    # Simulate an image-like input (3x3)
    img = np.random.rand(3, 3)
    # Simulate gradients w.r.t a specific target class
    grads = np.array([
        [0.0, 0.1, 0.0],
        [0.8, 0.9, 0.8],
        [0.0, 0.1, 0.0]
    ])
    
    s_map = get_saliency_map(img, grads)
    print("Saliency Map (importance of each pixel):")
    print(s_map)
