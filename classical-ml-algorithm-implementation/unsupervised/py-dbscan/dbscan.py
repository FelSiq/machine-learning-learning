import numpy as np
import scipy.spatial


class DBSCANFast:
    """Fast version of DBSCAN.

    Identity core, border and noise points, but does not separate into
    different clusters.
    """

    def __init__(
        self, radius: float, min_neighbors: int = 3, metric: str = "euclidean"
    ):
        assert float(radius) >= 0.0
        assert int(min_neighbors) > 0

        self.radius = float(radius)
        self.min_neighbors = int(min_neighbors)
        self.metric = str(metric)

        self.points_type = np.empty(0, dtype=int)

    def fit(self, X):
        X = np.asfarray(X)

        assert X.ndim == 2

        n = X.shape[0]

        dist = scipy.spatial.distance.cdist(X, X, metric=self.metric)
        neighbors = dist <= self.radius
        neighbors_num = neighbors.astype(int, copy=False).sum(axis=1)

        points_core_inds = np.flatnonzero(neighbors_num >= self.min_neighbors)
        points_border_inds = np.flatnonzero(
            np.any(neighbors[:, points_core_inds], axis=1)
        )

        self.points_type = np.zeros(n, dtype=np.uint8)
        self.points_type[points_border_inds] = 1
        self.points_type[points_core_inds] = 2

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.points_type


def _test():
    import matplotlib.pyplot as plt
    import sklearn.cluster

    n = 300

    X = np.vstack(
        (
            np.random.randn(n, 2),
            10 * np.random.randn(8, 2),
        )
    )

    radius = 0.5
    min_samples = 5

    ref = sklearn.cluster.DBSCAN(eps=radius, min_samples=3)
    y_preds_ref = (ref.fit_predict(X) >= 0).astype(int, copy=False)

    model = DBSCANFast(radius, 3)
    y_preds = model.fit_predict(X)

    colors = {0: "r", 1: "b", 2: "black"}

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.set_title("Mine")
    ax2.set_title("Reference (Sklearn)")

    ax1.scatter(*X.T, c=list(map(colors.get, y_preds)))
    ax2.scatter(*X.T, c=list(map(colors.get, y_preds_ref)))

    plt.show()


if __name__ == "__main__":
    _test()
