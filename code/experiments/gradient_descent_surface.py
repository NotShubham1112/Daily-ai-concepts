"""
Gradient Descent on Loss Surface
--------------------------------
Minimizing f(x) = (x - 3)^2
"""

x = 10.0
lr = 0.2

for step in range(15):
    grad = 2 * (x - 3)
    x -= lr * grad
    print(f"Step {step}: x={x:.4f}")
