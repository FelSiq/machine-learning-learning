"""Kernelized k-means."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def kernel_dot(x_a, x_b):
    return np.dot(x_a, x_b)


def kernel_gaussian(x_a, x_b, cov=2):
    return scipy.stats.multivariate_normal.pdf(x_a - x_b, mean=[0, 0], cov=cov)

kernel = kernel_gaussian

X1 = np.random.multivariate_normal([0, 0], 0.5 * np.eye(2), size=60)
X2 = np.random.multivariate_normal([0, 0], [[40, 0.2], [0.2, 40]], size=220)
X2 = np.array([v for v in X2 if np.linalg.norm(v) >= 8])

print("C1:", X1.size, "C2:", X2.size)

X = np.vstack((X1, X2))

m = X.shape[0]

plt.subplot(1, 2, 1)
plt.title("Data with real clusters")
plt.scatter(*X.T, c=np.repeat([0, 1], [X1.shape[0], X2.shape[0]]))

num_clusters = 2

clusters = np.random.randint(num_clusters, size=m, dtype=np.uint)

for it in np.arange(10):
    print(f"it: {it}...", end=" - ")

    clust_norm_sqr = np.zeros(num_clusters)

    cls_inds = []

    for k in np.arange(num_clusters):
        cls_inds.append(np.flatnonzero(clusters == k))

        for a in cls_inds[k]:
            for b in cls_inds[k]:
                clust_norm_sqr[k] += kernel(X[a, :], X[b, :])

        clust_norm_sqr[k] /= cls_inds[k].size * cls_inds[k].size

    diffs = 0

    for i in np.arange(m):
        new_clust_penalty = np.inf
        prev_cluster = clusters[i]

        for k in np.arange(num_clusters):
            cumsum = 0

            for j in cls_inds[k]:
                cumsum += kernel(X[i, :], X[j, :])

            cumsum /= cls_inds[k].size

            penalty = clust_norm_sqr[k] - 2 * cumsum

            if penalty < new_clust_penalty:
                clusters[i] = k
                new_clust_penalty = penalty

        diffs += int(prev_cluster != clusters[i])

    print(f"changed: {100 * diffs / m:.2f}%.   ", end="\r")

    if diffs / m < 0.0025:
        print("\nConverged (very few instances changed its cluster id).")
        break


print()

plt.subplot(1, 2, 2)
plt.title("Assigned clusters")
plt.scatter(*X.T, c=clusters)


plt.show()
