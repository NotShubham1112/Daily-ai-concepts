import numpy as np
import matplotlib.pyplot as plt

class Distributions:

    @staticmethod
    def gaussian(mu=0.0, sigma=1.0, n=10000):
        return np.random.normal(mu, sigma, n)

    @staticmethod
    def bernoulli(p=0.5, n=10000):
        return np.random.binomial(1, p, n)

    @staticmethod
    def poisson(lam=3.0, n=10000):
        return np.random.poisson(lam, n)


def visualize():
    np.random.seed(42)

    data = {
        "Gaussian": Distributions.gaussian(0, 1),
        "Bernoulli": Distributions.bernoulli(0.7),
        "Poisson": Distributions.poisson(4)
    }

    plt.figure(figsize=(12, 4))

    for i, (name, samples) in enumerate(data.items(), 1):
        plt.subplot(1, 3, i)
        plt.hist(samples, bins=50, density=True)
        plt.title(name)
        plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
