import typing as t

import numpy as np
import scipy.spatial


class LOF:
    def __init__(
        self, k: int, metric: str = "minkowski", contamination: t.Optional[float] = 1.5
    ):
        k = int(k)
        contamination = float(contamination) if contamination is not None else 1.5

        assert k > 0
        assert contamination > 0.0

        self.k = k
        self.metric = str(metric)
        self.lof = np.empty(0, dtype=float)
        self.contamination = contamination

    def fit(self, X):
        X = np.asfarray(X)

        dists = scipy.spatial.distance.cdist(X, X, metric=self.metric)
        nn_inds = np.argsort(dists, axis=0)[1 : 1 + self.k, :]
        farthest_nn_ind = nn_inds[-1, :]
        farthest_nn_dist = np.take_along_axis(
            dists, indices=np.expand_dims(farthest_nn_ind, 0), axis=0
        )

        reach_dist = np.maximum(farthest_nn_dist.T, dists)
        reach_dist = np.take_along_axis(reach_dist, nn_inds, axis=0)

        # LRD: Local Reachability Density
        lrd = 1.0 / (1e-10 + np.mean(reach_dist, axis=0))

        # LOF: Local Outlier Factor
        self.lof = np.mean(lrd[nn_inds], axis=0) / lrd

        return self

    def fit_predict(self, X):
        self.fit(X)
        return 2 * (self.lof < self.contamination).astype(int, copy=False) - 1


def _test():
    import sklearn.neighbors
    import sklearn.datasets

    X, _ = sklearn.datasets.load_iris(return_X_y=True)

    k = 20
    metric = "euclidean"

    ref_model = sklearn.neighbors.LocalOutlierFactor(n_neighbors=k, metric=metric)
    model = LOF(k=k, metric=metric)

    ref_cls = ref_model.fit_predict(X)
    cls = model.fit_predict(X)

    print(ref_cls)
    print(cls)

    assert np.allclose(ref_cls, cls)


if __name__ == "__main__":
    _test()
