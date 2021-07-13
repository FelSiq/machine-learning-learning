"""Novelty detection: semi-supervised anomaly detection.

Train dataset: clean.
Unseen dataset: possible dirty.
"""
import numpy as np
import sklearn.svm
import sklearn.covariance
import sklearn.ensemble
import sklearn.neighbors


contamination = 0.005

model_b = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=10,
    contamination=contamination,
    novelty=True,
)

model_d = sklearn.svm.OneClassSVM(nu=contamination)


import sklearn.datasets

n_in = 500

np.random.seed(16)

X = np.random.randn(n_in, 2)
X[: int(n_in * 0.2), :] += [8, 3]

y_preds_b = model_b.fit(X)
y_preds_d = model_d.fit(X)

n_in_test = 10
n_out = int(np.ceil(n_in * contamination))
X_test = np.vstack((10 * np.random.randn(n_out, 2), np.random.randn(n_in_test, 2)))

y_preds_b = model_b.predict(X_test).astype(int)
y_preds_d = model_d.predict(X_test).astype(int)


import matplotlib.pyplot as plt

plot_colors = {1: "black", -1: "red"}


fig, axes = plt.subplots(1, 3, figsize=(15, 10))
for i in range(3):
    axes[i].scatter(*X.T, c="blue")

axes[1].scatter(*X_test.T, s=128, marker="x", c=list(map(plot_colors.get, y_preds_b)))
axes[2].scatter(*X_test.T, s=128, marker="x", c=list(map(plot_colors.get, y_preds_d)))

axes[0].set_title("Original")
axes[1].set_title("LocalOutlierFactor")
axes[2].set_title("OneClassSVM")

plt.show()
