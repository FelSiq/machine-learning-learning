"""Principal Component Analysis (PCA) with SVD."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import sklearn.base


class PCAWithSVD(sklearn.base.TransformerMixin):
    def __init__(self, n_components: t.Union[int, float] = -1):
        assert not np.isclose(n_components, 0)

        self._mean = np.empty(0)
        self.loadings = np.empty(0)
        self.var_explained = np.empty(0)
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asfarray(X)
        self._mean = np.mean(X, axis=0)
        X -= self._mean
        n, m = X.shape

        if n > m:
            # (m, n) . (n, m) = (m, m)
            cov_mat = np.dot(X.T, X)
            D, V = np.linalg.eigh(cov_mat)
            D = np.flip(D)
            V = np.fliplr(V)

        else:
            # (n, m) . (m, n) = (n, n)
            cov_mat = np.dot(X, X.T)
            D, U = np.linalg.eigh(cov_mat)
            U = np.fliplr(U)
            D = np.flip(D)
            S = np.diag(np.sqrt(D))
            V = np.dot(U.T, np.dot(1.0 / S, X))

        if 0 < self.n_components < 1:
            self.n_components = int(np.ceil(m * self.n_components))

        self.n_components = max(self.n_components, 1)

        self.var_explained = D[: self.n_components :]
        self.loadings = V[:, : self.n_components]

        return self

    def transform(self, X):
        return np.dot(X, self.loadings)

    def scree_plot(self, ax=None, show: bool = True):
        if ax is None:
            ax = plt.gca()

        x = np.arange(1, 1 + self.n_components)
        var_prop = self.var_explained / np.sum(self.var_explained)
        ax.bar(x, height=var_prop, tick_label=x)

        if show:
            plt.show()

        return ax


def _test():
    def rm(angle):
        cos = np.cos(angle)
        sin = np.sin(angle)

        mat = np.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

        return mat

    np.random.seed(16)

    std = np.asfarray([1, 4])

    X = std * np.random.randn(200, 2)
    X = X @ rm(np.pi / 4)
    X -= np.mean(X, axis=0)

    model = PCAWithSVD(n_components=2)
    X_proj = model.fit_transform(X)

    aux = 3 * std * np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    aux_coords = model.transform(aux)

    if X_proj.shape[1] == 1:
        X_proj = np.column_stack((X_proj, np.zeros(X_proj.size)))
        aux_coords = np.column_stack((aux_coords, np.zeros(aux_coords.size)))

    plt.scatter(*X.T, label="original")
    plt.scatter(*X_proj.T, label="projected")
    plt.plot(*aux_coords[:2, :].T, linestyle="--", color="black")
    plt.plot(*aux_coords[2:, :].T, linestyle="--", color="black")

    plt.legend()
    plt.show()

    model.scree_plot()


if __name__ == "__main__":
    _test()
