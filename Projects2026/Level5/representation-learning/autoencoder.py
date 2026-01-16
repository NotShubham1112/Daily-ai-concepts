import numpy as np

class SimpleAutoencoder:
    def __init__(self, input_dim, latent_dim):
        # Encoder
        self.W_enc = np.random.randn(input_dim, latent_dim) * 0.1
        self.b_enc = np.zeros(latent_dim)
        
        # Decoder
        self.W_dec = np.random.randn(latent_dim, input_dim) * 0.1
        self.b_dec = np.zeros(input_dim)
        
    def encode(self, x):
        return np.tanh(x @ self.W_enc + self.b_enc)
        
    def decode(self, z):
        return z @ self.W_dec + self.b_dec # Linear reconstruction
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
