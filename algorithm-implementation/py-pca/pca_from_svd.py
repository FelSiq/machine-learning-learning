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
        self.n_components = n_components

        self.total_sing_vals = 0.0
        self.sing_vals = np.empty(0)

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
            S = np.sqrt(D)
            V = np.fliplr(V)

        else:
            # (n, m) . (m, n) = (n, n)
            cov_mat = np.dot(X, X.T)
            D, U = np.linalg.eigh(cov_mat)
            U = np.fliplr(U)
            D = np.flip(D)
            S = np.sqrt(D)
            V = np.dot(U.T, np.dot(1.0 / np.diag(S), X))

        self.total_sing_vals = float(np.sum(S))

        if 0 < self.n_components < 1:
            cumsum_sing_vals = np.cumsum(S) / self.total_sing_vals
            self.n_components = (
                1 + np.flatnonzero(cumsum_sing_vals >= self.n_components)[0]
            )

        self.n_components = max(int(self.n_components), 1)

        self.sing_vals = S[: self.n_components :]
        self.loadings = V[:, : self.n_components]

        self.total_var_explained = float(np.sum(self.sing_vals) / self.total_sing_vals)

        return self

    def transform(self, X):
        return np.dot(X, self.loadings)

    def scree_plot(self, fig=None, index=(121, 122)):
        if fig is None:
            fig = plt.figure()

        ax1 = fig.add_subplot(index[0])
        ax2 = fig.add_subplot(index[1])

        x = np.arange(1, 1 + self.n_components)
        var_prop = self.sing_vals / self.total_sing_vals

        ax1.bar(x, height=var_prop, tick_label=x)
        ax2.semilogy(x, self.sing_vals)

        ax1.set_title(
            f"Cumulative singular values (total: {self.total_var_explained:.4f})"
        )
        ax2.set_title("Semilog singular values")

        return fig, (ax1, ax2)


def _test():
    import sklearn.datasets

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X /= np.std(X, axis=0, ddof=0)

    model = PCAWithSVD(n_components=0.95)
    X_proj = model.fit_transform(X)

    colors = {0: "r", 1: "g"}

    fig = plt.figure()
    ax1 = fig.add_subplot(131)

    if X_proj.shape[1] <= 3:
        ax1.scatter(*X_proj.T, color=list(map(colors.get, y)))

    model.scree_plot(fig=fig, index=(132, 133))

    plt.show()


if __name__ == "__main__":
    _test()
