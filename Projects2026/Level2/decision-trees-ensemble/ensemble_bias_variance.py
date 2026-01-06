import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
from random_forest import RandomForest

def generate_data(n=100):
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

X, y = generate_data()

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
tree_acc = (tree.predict(X) == y).mean()

forest = RandomForest(n_trees=20, max_depth=3)
forest.fit(X, y)
forest_acc = (forest.predict(X) == y).mean()

print("Decision Tree Accuracy:", tree_acc)
print("Random Forest Accuracy:", forest_acc)
