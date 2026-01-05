import numpy as np

def mutual_information(x, y, bins=20):
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / np.sum(joint_hist)

    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)

    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (px[i] * py[j])
                )
    return mi


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.3

    print("Mutual Information:", mutual_information(x, y))
