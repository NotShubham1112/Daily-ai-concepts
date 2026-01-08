import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        hs, ys = {}, {}
        hs[-1] = h_prev

        for t in range(len(inputs)):
            hs[t] = np.tanh(
                self.Wxh @ inputs[t] +
                self.Whh @ hs[t-1] + self.bh
            )
            ys[t] = self.Why @ hs[t] + self.by

        return ys, hs
