import numpy as np

class MaxPool:
    def __init__(self, size=2):
        self.size = size

    def forward(self, input):
        self.input = input
        c, h, w = input.shape
        out = np.zeros((c, h // 2, w // 2))

        for k in range(c):
            for i in range(0, h, 2):
                for j in range(0, w, 2):
                    out[k, i//2, j//2] = np.max(input[k, i:i+2, j:j+2])

        return out
