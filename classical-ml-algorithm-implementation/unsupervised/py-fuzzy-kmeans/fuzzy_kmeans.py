"""Also known as Fuzzy C-means."""
import typing as t

import numpy as np
import scipy.spatial


class FuzzyKMeans:
    def __init__(
        self,
        k: int,
        p: float = 2.0,
        max_iter: int = 128,
        init: str = "k-means++",
        tol: float = 1e-3,
        random_state: t.Optional[int] = None,
    ):
        p = float(p)
        k = int(k)
        max_iter = int(max_iter)

        assert p >= 1.0
        assert k > 0
        assert max_iter > 0
        assert init in {"k-means++", "uniform"}

        self.p = p
        self.k = k
        self.max_iter = max_iter
        self.tol = float(tol)

        self.it_ = 0

        self.random_state = random_state
        self.init = init

        self.centroids_ = np.empty((0, 0), dtype=float)

    def _init_centroids(self, X):
        n, m = X.shape
        random_gen = np.random.default_rng(self.random_state)

        if self.init == "uniform" or self.k >= n:
            inds = random_gen.choice(n, size=min(self.k, n), replace=False)
            self.centroids_ = X[inds, :]
            return

        f_dist = lambda x, y: scipy.spatial.distance.cdist(
            x, y[np.newaxis, :], metric="sqeuclidean"
        )

        self.centroids_ = np.empty((self.k, m), dtype=float)
        first_centroid_id = random_gen.choice(n)
        self.centroids_[0, :] = X[first_centroid_id, :]
        closest_centroid_dist = f_dist(X, self.centroids_[0, :])

        for i in np.arange(1, self.k):
            p = closest_centroid_dist / float(np.sum(closest_centroid_dist))
            new_centroid_id = random_gen.choice(n, p=np.squeeze(p))
            self.centroids_[i, :] = X[new_centroid_id, :]

            if i >= self.k - 1:
                break

            new_centroid_dist = f_dist(X, self.centroids_[i, :])
            np.minimum(
                closest_centroid_dist, new_centroid_dist, out=closest_centroid_dist
            )

    def fit(self, X):
        X = np.asfarray(X)

        self._init_centroids(X)

        for i in np.arange(self.max_iter):
            # E-step
            W = self.predict(X)

            # M-step
            # W.T: (k, n)
            # X: (m, n)
            # W.T @ X: (k, n) @ (n, m): (k, m)
            prev_centroids = self.centroids_
            self.centroids_ = W.T @ X / np.sum(W.T, axis=1, keepdims=True)
            stop_criterion = float(np.max(np.abs(prev_centroids - self.centroids_)))
            if stop_criterion < self.tol:
                break

        self.it_ = i

        return self

    def predict(self, X):
        X = np.asfarray(X)
        dists = scipy.spatial.distance.cdist(X, self.centroids_, metric="sqeuclidean")
        np.power(dists, 1.0 / (self.p - 1.0), out=dists)
        np.maximum(dists, 1e-10, out=dists)
        # dists: (n, k)

        W = np.sum(1.0 / dists, axis=1, keepdims=True)
        # W: (n, 1)
        W = np.power(dists * W, -self.p)
        # W: (n, k)

        np.clip(W, 0.0, 1.0, out=W)

        return W

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


def _test():
    import matplotlib.pyplot as plt
    import sklearn.datasets

    k = 3

    X, y = sklearn.datasets.make_blobs(centers=k, random_state=16)
    np.random.seed(32)
    X += np.random.randn(*X.shape)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

    for i, p in enumerate((1.1, 1.5, 2, 3, 4, 5)):
        model = FuzzyKMeans(k=k, p=p, random_state=16, init="k-means++")
        y_preds = model.fit_predict(X)
        ax = axes[i // 3][i % 3]
        ax.set_title(f"$p={p}$ (it={model.it_})")
        ax.scatter(*X.T, c=y_preds)

    plt.show()


if __name__ == "__main__":
    _test()
