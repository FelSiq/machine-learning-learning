import numpy as np
import scipy.stats
import sklearn.base


class FactorAnalysis(sklearn.base.TransformerMixin):
    def __init__(self, n_components: int = 3, max_iter: int = 200):
        assert int(n_components) >= 1
        assert int(max_iter) >= 1

        self.mu = np.empty(0)
        self.cov = np.empty(0)

        self.L = np.empty(0)
        self.psi = np.empty(0)

        self.n_components = int(n_components)
        self.max_iter = int(max_iter)

    def _e_step(self, X_shifted, z):
        aux = self.L.T @ np.linalg.inv(self.L @ self.L.T + self.psi)

        self.mu_z = aux @ X_shifted

        self.cov_z = np.eye(aux.shape[0]) - aux @ self.L

        self.q = np.array(
            [
                scipy.stats.multivariate_normal.pdf(
                    z[:, i], mean=self.mu_z[:, i], cov=self.cov_z
                )
                for i in np.arange(z.shape[1])
            ]
        )

    def _m_step(self, X_shifted, X):
        m, n = X_shifted.shape
        k, _ = self.mu_z.shape
        L_aux_a = np.einsum("ji,ki->jk", X_shifted, self.mu_z)
        L_aux_b = np.einsum("ji,ki->jk", self.mu_z, self.mu_z) + n * self.cov_z
        L = L_aux_a @ np.linalg.inv(L_aux_b)
        phi_aux_a = np.einsum("ij,ik->jk", X, X)
        phi_aux_b = -np.einsum("ij,ki->jk", X, self.mu_z) @ self.L.T
        phi_aux_c = self.L @ L_aux_b @ self.L.T
        phi = np.diag(phi_aux_a + phi_aux_b + phi_aux_b.T + phi_aux_c) / n

        self.phi = phi
        self.L = L

    def _init_random_parameters(self, X):
        n, m = X.shape
        self.mu = np.mean(X, axis=0)
        self.psi = np.diag(np.random.random(m))
        self.L = np.random.random((m, self.n_components))
        z = np.random.randn(self.n_components, n)
        return z

    def fit(self, X, y=None):
        X = np.asfarray(X)

        n, m = X.shape
        z = self._init_random_parameters(X)
        X_shifted = (X - self.mu).T

        for i in np.arange(self.max_iter):
            self._e_step(X_shifted, z)
            self._m_step(X_shifted, X)

        return self

    def transform(self, X, y=None):
        X = np.asfarray(X)
        proj = np.linalg.inv(self.L.T @ self.L) @ self.L.T @ (X - self.mu).T
        return proj.T


def _test():
    import sklearn.datasets
    import sklearn.decomposition
    import matplotlib.pyplot as plt

    X, _ = sklearn.datasets.load_iris(return_X_y=True)

    model = FactorAnalysis(n_components=2)
    ref = sklearn.decomposition.FactorAnalysis(n_components=2)

    X_transf = model.fit_transform(X)
    X_transf_ref = ref.fit_transform(X)

    print(X_transf.shape)
    print(X_transf_ref.shape)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.scatter(*X_transf.T)
    ax1.set_title("X transformed (mine)")

    ax2.scatter(*X_transf_ref.T)
    ax2.set_title("X transformed (sklearn)")

    plt.show()


if __name__ == "__main__":
    _test()
