import collections

import sklearn.base
import numpy as np
import scipy.spatial

import union_find


class MeanShift(sklearn.base.ClusterMixin):
    def __init__(
        self,
        radius: float,
        max_iter: int = 256,
        centroid_equivalence_threshold: float = 1e-4,
    ):
        assert float(radius) > 0.0
        assert int(max_iter) > 0
        assert float(centroid_equivalence_threshold) >= 0.0

        self.radius = float(radius)
        self.max_iter = int(max_iter)
        self.centroid_equivalence_threshold = float(centroid_equivalence_threshold)

        self.centroids = np.empty((0, 0), dtype=float)
        self.labels_ = np.empty(0, dtype=int)

    def _group_centroids(self):
        dists = scipy.spatial.distance.cdist(self.centroids, self.centroids)
        ids_a, ids_b = np.nonzero(dists < self.radius)
        centroids_grouped = [[] for i in range(len(self.centroids))]

        for a, b in zip(ids_a, ids_b):
            centroids_grouped[a].append(b)

        return centroids_grouped

    def _filter_equivalent_centroids(self):
        dists = scipy.spatial.distance.cdist(self.centroids, self.centroids)
        grouper = union_find.UnionFindRank(len(self.centroids))
        ids_a, ids_b = np.nonzero(dists < self.centroid_equivalence_threshold)

        for id_a, id_b in zip(ids_a, ids_b):
            grouper.union(id_a, id_b)

        self.centroids = self.centroids[grouper.unique_ids()]

    def fit(self, X, y=None):
        X = np.asfarray(X)
        self.centroids = np.copy(X)

        for i in np.arange(self.max_iter):
            grouped_centroids_ids = self._group_centroids()

            for centroid_id, nearest_points_ids in enumerate(grouped_centroids_ids):
                nearest_points = self.centroids[nearest_points_ids, ...]
                self.centroids[centroid_id, ...] = np.mean(nearest_points, axis=0)

        self._filter_equivalent_centroids()

        dists = scipy.spatial.distance.cdist(X, self.centroids)
        self.labels_ = np.argmin(dists, axis=-1)

        return self

    def predict(self, X):
        return self.labels_


def _test():
    import matplotlib.pyplot as plt
    import sklearn.datasets
    import sklearn.cluster

    X, _ = sklearn.datasets.make_blobs(
        n_samples=150, n_features=2, centers=[(1.0, 0.5), (-2.0, -1.5)]
    )

    max_iter = 256
    bandwidth = 1.0

    ref = sklearn.cluster.MeanShift(bandwidth=bandwidth, max_iter=max_iter)
    model = MeanShift(radius=bandwidth, max_iter=max_iter)

    ref_preds = ref.fit_predict(X)
    preds = model.fit_predict(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

    all_colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "black",
        4: "yellow",
        5: "orange",
        6: "purple",
    }
    ref_colors = [all_colors[p_i] for p_i in ref_preds]
    colors = [all_colors[p_i] for p_i in preds]

    ax1.scatter(*X.T, c=ref_colors)
    ax2.scatter(*X.T, c=colors)
    ax1.set_title("Reference (sklearn)")
    ax2.set_title("Mine")

    plt.show()


if __name__ == "__main__":
    _test()
