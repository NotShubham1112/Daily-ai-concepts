"""
Gradient Visualization
----------------------
Gradient of a simple function f(x) = x^2
"""

def f(x):
    return x ** 2

def gradient(x):
    return 2 * x

x = 5.0
lr = 0.1

for step in range(10):
    grad = gradient(x)
    x = x - lr * grad
    print(f"Step {step}: x={x:.4f}, f(x)={f(x):.4f}")
