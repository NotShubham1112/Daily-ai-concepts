from sklearn.cluster import KMeans
from kmeans.kmeans_from_scratch import KMeansScratch
from data.load_data import load_synthetic
import matplotlib.pyplot as plt

def compare():
    X, _ = load_synthetic()

    scratch = KMeansScratch(k=3)
    labels_scratch = scratch.fit(X)

    sklearn_labels = KMeans(n_clusters=3).fit_predict(X)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("K-Means (Scratch)")
    plt.scatter(X[:,0], X[:,1], c=labels_scratch)

    plt.subplot(1,2,2)
    plt.title("K-Means (sklearn)")
    plt.scatter(X[:,0], X[:,1], c=sklearn_labels)

    plt.show()

if __name__ == "__main__":
    compare()
