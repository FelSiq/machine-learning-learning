import typing as t

import numpy as np
import scipy.stats
import scipy.spatial


class GaussianMixture:
    def __init__(
        self,
        n: int = 1,
        max_iter: int = 128,
        stop_threshold: float = 1e-4,
        random_state: t.Optional[int] = None,
    ):
        assert int(n) > 0
        assert int(max_iter) > 0

        self.n = int(n)
        self.max_iter = int(max_iter)
        self.random_state = random_state

        self._gaussians = []
        self._reg_cov = np.empty(0)
        self._stop_threshold = stop_threshold
        self._prior_probs = np.empty(0)

    def _calc_stop_criterion(self, means: np.ndarray, covs: np.ndarray) -> bool:
        if not self._gaussians:
            return False

        w_mean = len(means) / (len(means) + covs[0].size)
        diff = w_mean * np.linalg.norm(
            [means[:, i] - self._gaussians[i].mean for i in range(self.n)]
        )
        diff += (1.0 - w_mean) * np.mean(
            [np.linalg.norm(covs[i] - self._gaussians[i].cov) for i in range(self.n)]
        )

        stop_criterion = diff < self._stop_threshold

        return stop_criterion

    def _update_dists(self, X, centroid_conf) -> bool:
        means = np.dot(X.T, centroid_conf) / np.sum(centroid_conf, axis=0)

        covs = [
            np.cov(X, ddof=0, rowvar=False, aweights=centroid_conf[:, i])
            + self._reg_cov
            for i in range(self.n)
        ]

        stop_criterion = self._calc_stop_criterion(means, covs)

        self._gaussians = [
            scipy.stats.multivariate_normal(mean=means[:, i], cov=covs[i])
            for i in range(self.n)
        ]

        self._prior_probs = np.mean(centroid_conf, axis=0)

        return stop_criterion

    def fit(self, X: np.ndarray):
        X = np.asfarray(X)

        if X.ndim == 1:
            X = np.expand_axis(X, axis=1)

        if self.random_state is None:
            np.random.seed(self.random_state)

        self._reg_cov = np.diag(np.full(X.shape[1], fill_value=1e-6))
        X_vars = np.var(X, axis=0) + 1e-6

        mean_inds = np.random.choice(X.shape[0], size=self.n, replace=False)
        means = X[mean_inds, :].T

        self._prior_probs = np.full(self.n, fill_value=1.0 / self.n)

        self._gaussians = [
            scipy.stats.multivariate_normal(mean=means[:, i], cov=X_vars)
            for i in range(self.n)
        ]

        for _ in np.arange(self.max_iter):
            centroid_conf = self.predict(X)
            early_stop = self._update_dists(X, centroid_conf)

            if early_stop:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        centroid_conf = np.zeros((X.shape[0], self.n), dtype=float)

        for i in range(self.n):
            dist = self._gaussians[i]
            centroid_conf[:, i] = self._prior_probs[i] * dist.pdf(X)

        centroid_conf /= np.sum(centroid_conf, axis=1, keepdims=True)

        return centroid_conf


def _test():
    import matplotlib.pyplot as plt
    import sklearn.mixture
    import sklearn.datasets
    import sklearn.decomposition
    import sklearn.preprocessing

    n = 3
    X, y = sklearn.datasets.load_wine(return_X_y=True)

    model = GaussianMixture(n=n)
    model.fit(X)
    y_preds_my = model.predict(X)

    ref = sklearn.mixture.GaussianMixture(n_components=n)
    ref.fit(X)
    y_preds_ref = ref.predict(X)

    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    pca = sklearn.decomposition.PCA(n_components=2)
    X = pca.fit_transform(X)

    color_map = np.asarray(["r", "g", "b"])

    colors_orig = color_map[y]
    colors_my = color_map[np.argmax(y_preds_my, axis=1)]
    colors_ref = color_map[y_preds_ref]

    fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(10, 10))

    ax0.scatter(*X.T, c=colors_orig)
    ax0.set_title("Original")

    ax1.scatter(*X.T, c=colors_my)
    ax1.set_title("Mine")

    ax2.scatter(*X.T, c=colors_ref)
    ax2.set_title("Sklearn")

    plt.show()


if __name__ == "__main__":
    _test()
