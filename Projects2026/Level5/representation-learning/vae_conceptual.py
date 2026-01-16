import numpy as np

class VAEConceptual:
    """
    Conceptual implementation of VAE bottleneck.
    Focuses on the reparameterization trick and KL divergence.
    """
    def __init__(self, input_dim, latent_dim):
        # Encoder outputs mean and log_var
        self.W_mu = np.random.randn(input_dim, latent_dim) * 0.1
        self.W_var = np.random.randn(input_dim, latent_dim) * 0.1
        
    def encode(self, x):
        mu = x @ self.W_mu
        log_var = x @ self.W_var
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std
        
    def kl_divergence(self, mu, log_var):
        # KL loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        return -0.5 * np.sum(1 + log_var - np.square(mu) - np.exp(log_var))

if __name__ == "__main__":
    vae = VAEConceptual(10, 2)
    x = np.random.randn(5, 10)
    mu, log_var = vae.encode(x)
    z = vae.reparameterize(mu, log_var)
    kl = vae.kl_divergence(mu, log_var)
    
    print(f"Latent samples shape: {z.shape}")
    print(f"KL Divergence: {kl:.4f}")
