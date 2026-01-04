import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)

    p = gaussian_pdf(x, 0, 1)
    q = gaussian_pdf(x, 1, 1.5)

    p /= np.sum(p)
    q /= np.sum(q)

    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    plt.plot(x, p, label="P(x)")
    plt.plot(x, q, label="Q(x)")
    plt.title(f"KL(P||Q)={kl_pq:.4f} | KL(Q||P)={kl_qp:.4f}")
    plt.legend()
    plt.grid()
    plt.show()
