import numpy as np
import scipy.spatial


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 256, metric: str = "euclidean"):
        assert int(n_clusters) >= 1
        assert int(max_iter) >= 0

        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.metric = str(metric)

        self.cluster_ids = np.empty(0, dtype=int)
        self.centroids = np.empty(0, dtype=int)

    def _update_clusters(self, X):
        distances = scipy.spatial.distance.cdist(X, self.centroids, metric=self.metric)
        self.cluster_ids = distances.argmin(axis=1)
        return self.cluster_ids

    def fit(self, X):
        X = np.asfarray(X)

        assert X.ndim == 2

        n = X.shape[0]

        self.centroids = X[np.random.choice(n, size=self.n_clusters, replace=False), :]
        self.cluster_ids = self._update_clusters(X)

        rem_it = self.max_iter
        stop = bool(rem_it)

        while not stop:
            self._update_clusters(X)

            for k in np.arange(self.n_clusters):
                X_cluster = X[cluster_ids == k, :]
                self.centroids[k] = np.mean(X_cluster, axis=0)

            rem_it -= 1
            stop = bool(rem_it)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.cluster_ids


def _test():
    import matplotlib.pyplot as plt

    X = np.vstack([
        np.random.randn(40, 2) - (2, 2),
        4 * np.random.randn(40, 2) + (2, -5),
        16 * np.random.randn(40, 2) + (10, 0)
    ])

    model = KMeans(n_clusters=3)
    y_preds = model.fit_predict(X)

    colors = {0: "r", 1: "b", 2: "black"}

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.scatter(*X.T, c=list(map(colors.get, y_preds)))
    plt.show()


if __name__ == "__main__":
    _test()
