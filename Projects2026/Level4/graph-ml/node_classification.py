import numpy as np
from graph_convolutions import GCNLayer, normalize_adjacency, relu

def run_node_classification():
    # 1. Create a simple graph (2 communities)
    # Nodes 0-4 are community A, 5-9 are community B
    N = 10
    A = np.zeros((N, N))
    A[:5, :5] = 1 # Intra-community A
    A[5:, 5:] = 1 # Intra-community B
    A[0, 5] = A[5, 0] = 1 # One bridge edge
    
    A_norm = normalize_adjacency(A)
    
    # 2. Random features
    X = np.random.randn(N, 8)
    
    # 3. Simple 2-layer GCN
    layer1 = GCNLayer(8, 4)
    layer2 = GCNLayer(4, 2)
    
    # Forward Pass
    h1 = relu(layer1.forward(X, A_norm))
    logits = layer2.forward(h1, A_norm)
    
    print("Node Embeddings (Final Layer Logits):")
    print(logits)
    
    # Simple prediction based on max logit
    preds = np.argmax(logits, axis=1)
    print("\nPredicted Classes:", preds)

if __name__ == "__main__":
    run_node_classification()
