"""Outlier detection: unsupervised anomaly detection.

Train dataset: dirty.
Unseen dataset: not even considered.
"""
import numpy as np
import sklearn.svm
import sklearn.covariance
import sklearn.ensemble
import sklearn.neighbors


contamination = 0.005

model_a = sklearn.ensemble.IsolationForest(
    n_estimators=250, bootstrap=True, contamination=contamination, warm_start=False
)

model_b = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=10, contamination=contamination
)


# Note: only works we data is unimodal (i.e. only a single dense
# region)
model_c = sklearn.covariance.EllipticEnvelope(contamination=contamination)


# Note: bad if not fine tuned properly. It is more suitable for
# novelty detection (whereas the train dataset is assumed to be
# clean, and we are worried in detecting outliers only in new,
# unseen data.
model_d = sklearn.svm.OneClassSVM(nu=contamination)


import sklearn.datasets

n_in = 500
n_out = int(np.ceil(n_in * contamination))

np.random.seed(16)

X = np.vstack((np.random.randn(n_in, 2), 10 * np.random.randn(n_out, 2)))
X[: int(n_in * 0.2), :] += [8, 3]
y = np.concatenate((np.zeros(n_in), np.ones(n_out)))
inds = np.arange(y.size)
np.random.shuffle(inds)


y_preds_a = model_a.fit_predict(X).astype(int)
y_preds_b = model_b.fit_predict(X).astype(int)
y_preds_c = model_c.fit_predict(X).astype(int)
y_preds_d = model_d.fit_predict(X).astype(int)


import matplotlib.pyplot as plt

plot_colors = {0: "blue", 1: "red"}


fig, axes = plt.subplots(1, 5, figsize=(15, 10))
for i in range(5):
    axes[i].scatter(*X.T, c=list(map(plot_colors.get, y)))

axes[1].scatter(*X[y_preds_a == -1, :].T, s=128, marker="x", c="black")
axes[2].scatter(*X[y_preds_b == -1, :].T, s=128, marker="x", c="black")
axes[3].scatter(*X[y_preds_c == -1, :].T, s=128, marker="x", c="black")
axes[4].scatter(*X[y_preds_d == -1, :].T, s=128, marker="x", c="black")

axes[0].set_title("Original")
axes[1].set_title("Isolation Forest")
axes[2].set_title("LocalOutlierFactor")
axes[3].set_title("EllipiticEnvelope")
axes[4].set_title("OneClassSVM")

plt.show()
