import matplotlib.pyplot as plt

def plot_clusters(X, labels, title="Clusters"):
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title(title)
    plt.show()
