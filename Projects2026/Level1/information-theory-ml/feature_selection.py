import numpy as np
from entropy_metrics import entropy

def information_gain(feature, target, bins=10):
    joint_hist, _, _ = np.histogram2d(feature, target, bins=bins)
    joint_prob = joint_hist / np.sum(joint_hist)

    p_feature = np.sum(joint_prob, axis=1)
    p_target = np.sum(joint_prob, axis=0)

    h_target = entropy(p_target)
    conditional_entropy = 0.0

    for i in range(len(p_feature)):
        if p_feature[i] > 0:
            conditional_entropy += p_feature[i] * entropy(
                joint_prob[i] / p_feature[i]
            )

    return h_target - conditional_entropy


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    y = (X[:, 0] + np.random.randn(1000) * 0.2 > 0).astype(int)

    for i in range(X.shape[1]):
        ig = information_gain(X[:, i], y)
        print(f"Feature {i} Information Gain: {ig:.4f}")
