import numpy as np
import scipy.spatial


class SphericalKMeans:
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
        X = np.copy(X).astype(float, copy=False)

        assert X.ndim == 2

        X /= 1e-10 + np.linalg.norm(X, axis=1, ord=2, keepdims=True)

        self._init_centroids(X)
        self.cluster_ids = self._update_clusters(X)

        rem_it = self.max_iter
        stop = bool(rem_it)

        while not stop:
            self._update_clusters(X)

            for k in np.arange(self.n_clusters):
                X_cluster = X[cluster_ids == k, :]
                new_centroid = np.mean(X_cluster, axis=0)
                new_centroid /= 1e-10 + np.linalg.norm(new_centroid, ord=2)
                self.centroids[k] = new_centroid

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

    for i in range(m):
        X = np.vstack(
            [
                np.random.randn(40, 2) - (2, 2),
                4 * np.random.randn(40, 2) + (2, -5),
                16 * np.random.randn(40, 2) + (10, 0),
            ]
        )
        X = (
            2
            * (X - np.min(X, axis=0, keepdims=True))
            / (1e-10 + np.ptp(X, axis=0, keepdims=True))
            - 1
        )
        X_norm = X / (1e-10 + np.linalg.norm(X, axis=1, keepdims=True, ord=2))

        model_rand = SphericalKMeans(n_clusters=3, init="random")
        model_pp = SphericalKMeans(n_clusters=3, init="kmeans++")

        y_preds_rand = model_rand.fit_predict(X)
        y_preds_pp = model_pp.fit_predict(X)

        colors = {0: "r", 1: "b", 2: "black"}

        ax = axes[0][i % m]
        c = list(map(colors.get, y_preds_rand))
        ax.scatter(*X.T, c=c)
        ax.scatter(*X_norm.T, c=c, marker="+")
        ax.set_title("Random")

        ax = axes[1][i % m]
        c = list(map(colors.get, y_preds_pp))
        ax.scatter(*X.T, c=c)
        ax.scatter(*X_norm.T, c=c, marker="+")
        ax.set_title("K-means++")

    plt.show()


if __name__ == "__main__":
    _test()
