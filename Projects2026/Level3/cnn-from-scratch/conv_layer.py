import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / filter_size

    def forward(self, input):
        self.input = input
        h, w = input.shape
        f = self.filter_size
        output_dim = h - f + 1
        output = np.zeros((self.num_filters, output_dim, output_dim))

        for k in range(self.num_filters):
            for i in range(output_dim):
                for j in range(output_dim):
                    region = input[i:i+f, j:j+f]
                    output[k, i, j] = np.sum(region * self.filters[k])

        return output

    def backward(self, d_out, lr):
        d_filters = np.zeros_like(self.filters)

        for k in range(self.num_filters):
            for i in range(d_out.shape[1]):
                for j in range(d_out.shape[2]):
                    region = self.input[i:i+self.filter_size, j:j+self.filter_size]
                    d_filters[k] += d_out[k, i, j] * region

        self.filters -= lr * d_filters
