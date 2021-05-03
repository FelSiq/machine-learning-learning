import numpy as np
import scipy.stats
import sklearn.base


class FactorAnalysis(sklearn.base.TransformerMixin):
    def __init__(
        self, n_components: int = 3, max_iter: int = 1000, threshold: float = 1e-3
    ):
        assert int(n_components) >= 1
        assert int(max_iter) >= 1
        assert float(threshold) >= 0

        self.proj_mat = np.empty(0)

        self.mu = np.empty(0)
        self.cov_z = np.empty(0)
        self.mu_z = np.empty(0)

        self.L = np.empty(0)
        self.psi = np.empty(0)

        self.n_components = int(n_components)
        self.max_iter = int(max_iter)

        self.log_likelihood = 0.0
        self.threshold = float(threshold)

    def _e_step(self, X_shifted):
        aux = self.L.T @ np.linalg.inv(self.L @ self.L.T + np.diag(self.psi))
        self.mu_z = aux @ X_shifted
        self.cov_z = np.eye(aux.shape[0]) - aux @ self.L

    def _m_step(self, X_shifted):
        m, n = X_shifted.shape
        k, _ = self.mu_z.shape

        L_aux_a = np.einsum("ji,ki->jk", X_shifted, self.mu_z)
        L_aux_b = np.einsum("ji,ki->jk", self.mu_z, self.mu_z) + n * self.cov_z
        L = L_aux_a @ np.linalg.inv(L_aux_b)

        # NOTE: Those lecture notes: http://cs229.stanford.edu/summer2019/cs229-notes9.pdf
        # imply that the correct solution would be:
        # phi_aux_a = np.einsum("ji,ki->jk", X.T, X.T)
        # But this makes the result awfully weird visually.
        phi_aux_a = np.einsum("ji,ki->jk", X_shifted, X_shifted)
        phi_aux_b = -np.einsum("ji,ki->jk", X_shifted, self.mu_z) @ self.L.T
        phi_aux_c = self.L @ L_aux_b @ self.L.T
        phi = (phi_aux_a + phi_aux_b + phi_aux_b.T + phi_aux_c) / n
        psi = np.diag(phi)

        self.psi = psi
        self.L = L

    def _init_random_parameters(self, X):
        n, m = X.shape
        self.mu = np.mean(X, axis=0)
        self.psi = np.ones(m)
        self.L = np.random.randn(m, self.n_components)

    def _compute_lll(self, X):
        dist = scipy.stats.multivariate_normal(
            mean=self.mu, cov=self.L @ self.L.T + np.diag(self.psi)
        )
        lll = float(np.sum(dist.logpdf(X)))
        return lll

    def fit(self, X, y=None):
        X = np.asfarray(X)

        n, m = X.shape
        self._init_random_parameters(X)
        X_shifted = (X - self.mu).T
        self.log_likelihood = -np.inf

        for i in np.arange(self.max_iter):
            self._e_step(X_shifted)
            self._m_step(X_shifted)

            prev_log_likelihood = self.log_likelihood
            self.log_likelihood = self._compute_lll(X)

            if self.log_likelihood - prev_log_likelihood < self.threshold:
                break

        self.proj_mat = np.linalg.inv(self.L.T @ self.L) @ self.L.T

        return self

    def transform(self, X, y=None):
        X = np.asfarray(X)
        proj = self.proj_mat @ (X - self.mu).T
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
