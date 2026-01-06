from collections import defaultdict

def compute_feature_importance(tree):
    importance = defaultdict(int)

    def traverse(node):
        if node is None or node.value is not None:
            return
        importance[node.feature] += 1
        traverse(node.left)
        traverse(node.right)

    traverse(tree.root)
    return dict(importance)
