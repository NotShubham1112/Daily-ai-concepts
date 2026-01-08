import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))

    def step(self, x, h_prev, c_prev):
        concat = np.vstack((x, h_prev))
        z = self.W @ concat + self.b

        f, i, o, g = np.split(z, 4)
        f = 1 / (1 + np.exp(-f))
        i = 1 / (1 + np.exp(-i))
        o = 1 / (1 + np.exp(-o))
        g = np.tanh(g)

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        return h, c
