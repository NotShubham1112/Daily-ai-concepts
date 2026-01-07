import numpy as np
from conv_layer import Conv2D
from pooling import MaxPool

class SimpleCNN:
    def __init__(self):
        self.conv = Conv2D(num_filters=8, filter_size=3)
        self.pool = MaxPool()
        self.fc = np.random.randn(8 * 13 * 13, 10) / 100

    def forward(self, x):
        out = self.conv.forward(x)
        out = np.maximum(out, 0)  # ReLU
        out = self.pool.forward(out)
        self.flatten = out.flatten()
        logits = self.flatten @ self.fc
        return logits

    def backward(self, d_logits, lr):
        d_fc = np.outer(self.flatten, d_logits)
        self.fc -= lr * d_fc
