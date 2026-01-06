import umap
import matplotlib.pyplot as plt
from data.load_data import load_synthetic

def visualize():
    X, y = load_synthetic()
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X)

    plt.scatter(X_umap[:,0], X_umap[:,1], c=y)
    plt.title("UMAP Projection")
    plt.show()

if __name__ == "__main__":
    visualize()
