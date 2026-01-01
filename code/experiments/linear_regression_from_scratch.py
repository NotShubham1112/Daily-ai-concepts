"""
Linear Regression from Scratch
------------------------------
y = wx + b using gradient descent
"""

import random

# Data
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]

w = random.random()
b = random.random()
lr = 0.01

for epoch in range(1000):
    for x, y in zip(X, Y):
        y_pred = w * x + b
        error = y_pred - y

        w -= lr * error * x
        b -= lr * error

print("Learned weight:", w)
print("Learned bias:", b)
