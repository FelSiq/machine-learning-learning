import numpy as np
import scipy.spatial


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 256,
        init: str = "random",
    ):
        assert int(n_clusters) >= 1
        assert int(max_iter) >= 0
        assert init in {"random", "kmeans++"}

        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.init = init

        self.cluster_ids = np.empty(0, dtype=int)
        self.centroids = np.empty(0, dtype=int)

    def _update_clusters(self, X):
        distances = scipy.spatial.distance.cdist(X, self.centroids, metric="euclidean")
        self.cluster_ids = distances.argmin(axis=1)
        return self.cluster_ids

    def _init_centroids(self, X):
        n, m = X.shape

        if self.init == "random" or self.n_clusters == 1 or self.n_clusters >= n:
            self.centroids = X[
                np.random.choice(n, size=self.n_clusters, replace=False), :
            ]
            return

        # K-means++ initialization
        self.centroids = np.vstack(
            [
                X[np.random.choice(n, size=1), :],
                np.empty((self.n_clusters - 1, m), dtype=float),
            ]
        )

        centroid_min_dists = np.full(n, fill_value=np.inf, dtype=float)

        for k in np.arange(1, self.n_clusters):
            new_centroid_dist = scipy.spatial.distance.cdist(
                X, self.centroids[k - 1, None, :], metric="sqeuclidean"
            ).ravel()
            np.minimum(centroid_min_dists, new_centroid_dist, out=centroid_min_dists)

            sample_probs = centroid_min_dists / float(np.sum(centroid_min_dists))

            new_centroid_id = np.random.choice(n, size=1, p=sample_probs)
            self.centroids[k, :] = X[new_centroid_id, ...]

    def fit(self, X):
        X = np.asfarray(X)

        assert X.ndim == 2

        self._init_centroids(X)
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

    np.random.seed(16)

    m = 5
    fig, axes = plt.subplots(2, m, figsize=(15, 10))

    for i in range(10):
        X = np.vstack(
            [
                np.random.randn(40, 2) - (2, 2),
                4 * np.random.randn(40, 2) + (2, -5),
                16 * np.random.randn(40, 2) + (10, 0),
            ]
        )

        model = KMeans(n_clusters=3, init="random" if i < m else "kmeans++")
        y_preds = model.fit_predict(X)
        colors = {0: "r", 1: "b", 2: "black"}

        ax = axes[i // m][i % m]
        ax.scatter(*X.T, c=list(map(colors.get, y_preds)))
        ax.set_title("K-means" if i < m else "K-means++")

    plt.show()


if __name__ == "__main__":
    _test()
