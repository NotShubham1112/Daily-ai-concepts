import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer import MiniTransformerBlock
from positional_encoding import get_positional_encoding

def visualize_attention():
    # Setup
    seq_len = 10
    d_model = 64
    d_ff = 256
    
    # Input with PE
    x = np.random.randn(seq_len, d_model)
    pe = get_positional_encoding(seq_len, d_model)
    x = x + pe
    
    # Run Transformer Block
    transformer = MiniTransformerBlock(d_model, d_ff)
    output, weights = transformer.forward(x)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=True, cmap="viridis", 
                xticklabels=[f"Token {i}" for i in range(seq_len)],
                yticklabels=[f"Token {i}" for i in range(seq_len)])
    plt.title("Self-Attention Weights Heatmap")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()

if __name__ == "__main__":
    visualize_attention()
