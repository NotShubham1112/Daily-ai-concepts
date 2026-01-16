import numpy as np

def quantize_weights(weights):
    """
    Min-max quantization to Int8.
    """
    min_val = np.min(weights)
    max_val = np.max(weights)
    
    # Scale and Shift
    scale = (max_val - min_val) / 255.0
    zero_point = 0 # Simplified
    
    quantized = np.round((weights - min_val) / (scale + 1e-10)).astype(np.uint8)
    return quantized, scale, min_val

def dequantize_weights(quantized, scale, min_val):
    return (quantized.astype(float) * scale) + min_val

if __name__ == "__main__":
    w = np.random.randn(100)
    q_w, s, m = quantize_weights(w)
    de_w = dequantize_weights(q_w, s, m)
    
    error = np.mean(np.square(w - de_w))
    print(f"Quantization Error (MSE): {error:.6f}")
    print(f"Original size (float64): {w.nbytes} bytes")
    print(f"Quantized size (uint8):  {q_w.nbytes} bytes")
