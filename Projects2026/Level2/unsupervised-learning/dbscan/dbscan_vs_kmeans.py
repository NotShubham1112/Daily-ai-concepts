from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

def compare(X):
    km = KMeans(n_clusters=3).fit_predict(X)
    db = DBSCAN(eps=0.4).fit_predict(X)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("K-Means")
    plt.scatter(X[:,0], X[:,1], c=km)

    plt.subplot(1,2,2)
    plt.title("DBSCAN")
    plt.scatter(X[:,0], X[:,1], c=db)
    plt.show()
