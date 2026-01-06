from sklearn.datasets import make_blobs

def generate():
    X, y = make_blobs(
        n_samples=500,
        centers=3,
        cluster_std=1.2,
        random_state=42
    )
    return X, y
