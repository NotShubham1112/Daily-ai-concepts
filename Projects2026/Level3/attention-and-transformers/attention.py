import numpy as np
from utils import softmax

class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        # Initialize weights randomly (Xavier initialization-like)
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, X):
        """
        X: (seq_len, d_model)
        """
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        # Scaled Dot-Product Attention
        score = (Q @ K.T) / np.sqrt(self.d_model)
        attention_weights = softmax(score)
        
        output = attention_weights @ V
        return output, attention_weights
