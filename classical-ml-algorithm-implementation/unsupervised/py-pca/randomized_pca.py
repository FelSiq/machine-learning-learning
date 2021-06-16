import typing as t

import numpy as np
import sklearn.base
import matplotlib.pyplot as plt


class RandomizedPCA(sklearn.base.TransformerMixin):
    def __init__(
        self,
        n_components: t.Optional[int] = None,
        oversampling: int = 0,
        power_iterations: int = 3,
        copy: bool = True,
    ):
        assert int(n_components) >= 1
        assert int(oversampling) >= 0
        assert int(power_iterations) >= 0

        self.n_components = int(n_components)
        self.oversampling = int(oversampling)
        self.power_iterations = int(power_iterations)

        self.Q = np.empty(0)
        self.sing_vals = np.empty(0)
        self.loadings = np.empty(0)
        self.copy = copy

    def fit(self, X, y=None):
        if self.copy:
            X = np.copy(X).astype(float, copy=False)

        else:
            X = np.asfarray(X)

        n, m = X.shape

        self.mean = np.mean(X, axis=0)

        X -= self.mean

        n_components = self.n_components if self.n_components is not None else m

        P = np.random.randn(m, n_components + self.oversampling)
        # (n, m) (m, r)
        Z = np.dot(X, P)
        # Z: (n, r)

        for _ in np.arange(self.power_iterations):
            Z = np.dot(X, np.dot(X.T, Z))
            # (n, r) = (n, m) . (m, n) . (n, r)

        self.Q, _ = np.linalg.qr(Z, mode="reduced")

        return self

    def transform(self, X, y=None):
        # Q: (n, r)
        Y = np.dot(self.Q.T, X - self.mean)
        # Y: (r, n) . (n, m) = (r, m)
        Uy, self.sing_vals, self.loadings = np.linalg.svd(Y, full_matrices=False)

        self.sing_vals = self.sing_vals[: -self.oversampling]
        self.loadings = self.loadings[:, : -self.oversampling].T

        # Uy: (r, r)
        Ux = np.dot(self.Q, Uy)
        # Ux: (n, r) (r, r) = (n, r)
        Ux = Ux[:, : -self.oversampling]

        proj = np.dot(Ux, np.diag(self.sing_vals))

        return proj

    def scree_plot(self, X=None, fig=None, index=(121, 122)):
        if fig is None:
            fig = plt.figure()

        if X is not None:
            X = self.transform(X)

        ax1 = fig.add_subplot(index[0])
        ax2 = fig.add_subplot(index[1])

        x = np.arange(1, 1 + self.sing_vals.size)
        ax1.bar(x, self.sing_vals)
        ax2.semilogy(self.sing_vals)

        ax1.set_title("Singular values")
        ax2.set_title("Semilog singular values")

        return fig, (ax1, ax2), X


def _test():
    import sklearn.datasets
    import sklearn.decomposition

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X /= np.std(X, axis=0)

    oversampling = 2
    iterated_power = 10

    model = RandomizedPCA(
        n_components=2, oversampling=oversampling, power_iterations=iterated_power
    )
    ref = sklearn.decomposition.PCA(
        n_components=2 + oversampling,
        svd_solver="randomized",
        iterated_power=iterated_power,
    )

    X_proj = model.fit_transform(X)
    X_proj_ref = ref.fit_transform(X)[:, :-oversampling]

    fig = plt.figure(figsize=(10, 10))

    if X_proj.shape[1] == 2:
        ax1 = fig.add_subplot(131)
        ax1.scatter(*X_proj_ref.T, label="reference")
        ax1.scatter(*X_proj.T, label="mine")
        ax1.legend()

    model.scree_plot(fig=fig, index=(132, 133))

    plt.show()


if __name__ == "__main__":
    _test()
