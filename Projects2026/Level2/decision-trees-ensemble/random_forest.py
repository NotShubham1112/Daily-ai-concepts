import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, sample_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, X, y):
        n_samples = int(len(X) * self.sample_ratio)

        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
