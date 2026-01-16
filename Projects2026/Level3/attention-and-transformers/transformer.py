import numpy as np
from attention import SelfAttention
from utils import softmax

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
        
    def forward(self, x):
        return (self.relu(x @ self.W1 + self.b1)) @ self.W2 + self.b2

class MiniTransformerBlock:
    def __init__(self, d_model, d_ff):
        self.attention = SelfAttention(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
        
    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta
        
    def forward(self, x):
        # Attention Sub-layer
        att_out, weights = self.attention.forward(x)
        x = self.layer_norm(x + att_out, self.norm1_gamma, self.norm1_beta)
        
        # Feed-Forward Sub-layer
        ff_out = self.feed_forward.forward(x)
        x = self.layer_norm(x + ff_out, self.norm2_gamma, self.norm2_beta)
        
        return x, weights
