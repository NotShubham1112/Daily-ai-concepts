from sklearn.manifold import TSNE
from data.load_data import load_synthetic
import matplotlib.pyplot as plt

def visualize():
    X, y = load_synthetic()
    X_tsne = TSNE(n_components=2, perplexity=30).fit_transform(X)

    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)
    plt.title("t-SNE Projection")
    plt.show()

if __name__ == "__main__":
    visualize()
