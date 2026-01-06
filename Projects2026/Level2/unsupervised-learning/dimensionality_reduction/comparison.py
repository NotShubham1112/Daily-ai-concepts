from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

def reduce_and_plot(X, y=None):
    reducers = {
        "PCA": PCA(2),
        "t-SNE": TSNE(2),
        "UMAP": umap.UMAP(2)
    }

    plt.figure(figsize=(12,4))
    for i, (name, model) in enumerate(reducers.items()):
        X_red = model.fit_transform(X)
        plt.subplot(1,3,i+1)
        plt.title(name)
        plt.scatter(X_red[:,0], X_red[:,1], c=y)
    plt.show()
