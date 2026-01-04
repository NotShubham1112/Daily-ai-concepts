import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class BayesianBernoulli:
    def __init__(self, alpha=1, beta_=1):
        self.alpha = alpha
        self.beta = beta_

    def update(self, data):
        successes = np.sum(data)
        failures = len(data) - successes
        self.alpha += successes
        self.beta += failures

    def posterior(self, x):
        return beta.pdf(x, self.alpha, self.beta)


if __name__ == "__main__":
    np.random.seed(0)

    true_p = 0.7
    data = np.random.binomial(1, true_p, 50)

    model = BayesianBernoulli(alpha=1, beta_=1)
    model.update(data)

    x = np.linspace(0, 1, 500)
    posterior = model.posterior(x)

    plt.plot(x, posterior)
    plt.title("Posterior Distribution after Observations")
    plt.xlabel("Probability of Success")
    plt.ylabel("Density")
    plt.grid()
    plt.show()
