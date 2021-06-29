import itertools
import bisect

import numpy as np
import scipy.spatial


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.uint)
        self.rank = np.zeros(n, dtype=np.uint)
        self.n = int(n)

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)

        if px == py:
            return

        if self.rank[px] < self.rank[py]:
            self.parent[px] = py

        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px

        else:
            self.parent[px] = py
            self.rank[py] += 1


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 2, linkage: str = "ward"):
        assert int(n_clusters) >= 2
        assert linkage in {"ward", "single", "complete", "average"}

        self.linkage = linkage
        self.n_clusters = int(n_clusters)
        self._cluster_ids_ds = UnionFind(0)

        self.clusters = frozenset()
        self.cluster_ids = np.empty(0, dtype=int)

        if self.linkage == "single":
            self.linkage_fun = np.min

        elif self.linkage == "complete":
            self.linkage_fun = np.max

        elif self.linkage == "average":
            self.linkage_fun = np.mean

        else:
            self.linkage_fun = np.sum

    def _update_cluster_ids(self, shrink: bool = False):
        all_clust_ids_iter = map(
            self._cluster_ids_ds.find, range(self._cluster_ids_ds.n)
        )
        self.cluster_ids = np.fromiter(all_clust_ids_iter, dtype=int)
        self.clusters = frozenset(self.cluster_ids)

        if shrink:
            shrinked_ids_iter = map(
                lambda x: bisect.bisect_left(sorted(self.clusters), x), self.cluster_ids
            )
            self.cluster_ids = np.fromiter(shrinked_ids_iter, dtype=int)

    def fit(self, X):
        X = np.asfarray(X)

        assert X.ndim == 2

        n = len(X)
        self._cluster_ids_ds = UnionFind(n)

        it = 0
        total_it = n - self.n_clusters

        while it < total_it:
            champion_comb = None
            champion_comb_cost = np.inf

            self._update_cluster_ids()
            cache_insts_views = {
                cls_id: X[self.cluster_ids == cls_id, :] for cls_id in self.clusters
            }

            for cluster_a, cluster_b in itertools.combinations(self.clusters, 2):
                X_a = cache_insts_views[cluster_a]
                X_b = cache_insts_views[cluster_b]

                dists = scipy.spatial.distance.cdist(X_a, X_b, metric="sqeuclidean")
                cost = self.linkage_fun(dists)

                if cost < champion_comb_cost:
                    champion_comb_cost = cost
                    champion_comb = (cluster_a, cluster_b)

            self._cluster_ids_ds.union(*champion_comb)
            it += 1

        self._update_cluster_ids(shrink=True)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.cluster_ids


if __name__ == "__main__":
    import sklearn.cluster

    np.random.seed(16)

    n_clusters = 10
    linkage = "ward"

    n = 33

    X = np.vstack(
        [
            np.random.randn(n, 2),
            4 * np.random.randn(n, 2) + (-1, 2),
            2 * np.random.randn(n, 2) + (3, -4),
        ]
    )

    ref = sklearn.cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage
    )
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    y_preds = model.fit_predict(X)
    y_preds_ref = ref.fit_predict(X)

    print(y_preds)
    print(y_preds_ref)
