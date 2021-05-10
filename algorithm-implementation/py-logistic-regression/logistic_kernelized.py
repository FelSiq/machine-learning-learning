"""
Note: this same code can be used to any Generalized Linear Model, i.e.,
any statistical model that models an y that follows an distribution in the
exponential family.

The only change necessary is switching the sigmoid function to the appropriate
activation function that correctly describes the y distribution.
"""
import functools

import numpy as np
import scipy.spatial


class KernelLogReg:
    KERNEL_FUNC = {
        "gaussian": lambda dist_sqr, bandwidth=1.0: np.exp(
            -0.5 / bandwidth ** 2 * dist_sqr
        ),
    }

    def __init__(
        self,
        learning_rate: float = 1e-2,
        kernel: str = "gaussian",
        max_epochs: int = 256,
        add_intercept: bool = True,
        *args,
        **kwargs,
    ):
        assert float(learning_rate) > 0.0
        assert kernel in {"gaussian"}
        assert int(max_epochs) > 0

        self.kernel = kernel
        self._kernel_fun = self.KERNEL_FUNC[kernel]
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.add_intercept = add_intercept

        np.coeffs = np.empty(0)

        self._kernel_fun = functools.partial(self._kernel_fun, *args, **kwargs)

    def _calc_kernel_mat(self, A, B=None):
        if B is None:
            B = A

        dist_sqr = scipy.spatial.distance.cdist(A, B, metric="sqeuclidean")
        kernel_mat = self._kernel_fun(dist_sqr)

        return kernel_mat

    @staticmethod
    def _sigmoid(x):
        pos_inds = x >= 0
        neg_inds = ~pos_inds

        neg_exp = np.exp(x[neg_inds])

        res = np.zeros_like(x)

        res[pos_inds] = 1.0 / (1.0 + np.exp(-x[pos_inds]))
        res[neg_inds] = neg_exp / (1.0 + neg_exp)

        return res

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.asfarray(y).ravel()

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        self.X = X
        self.coeffs = np.empty(n)
        kernel_mat = self._calc_kernel_mat(self.X)

        for i in np.arange(self.max_epochs):
            preds = self._sigmoid(kernel_mat @ self.coeffs)
            update = self.learning_rate * (preds - y)
            self.coeffs -= update

        return self

    def predict(self, X):
        X = np.asfarray(X)

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        kernel_mat = self._calc_kernel_mat(X, self.X)
        preds = self._sigmoid(kernel_mat @ self.coeffs)

        return preds


def _test():
    import sklearn.metrics
    import sklearn.model_selection
    import matplotlib.pyplot as plt

    np.random.seed(16)

    n = 1000
    X = np.random.randn(n, 2)
    X_norm = np.linalg.norm(X, ord=2, axis=1)
    y = np.logical_or(np.logical_and(1 <= X_norm, X_norm <= 2), X_norm > 3).astype(
        int, copy=False
    )

    model = KernelLogReg(bandwidth=0.4, max_epochs=1000)

    def rmse(a, b):
        return sklearn.metrics.mean_squared_error(a, b, squared=False)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=16
    )

    model.fit(X_train, y_train)

    y_preds_train = (model.predict(X_train) > 0.5).astype(int, copy=False)
    y_preds_eval = (model.predict(X_eval) > 0.5).astype(int, copy=False)

    rmse_train = rmse(y_preds_train, y_train)
    rmse_eval = rmse(y_preds_eval, y_eval)

    print(f"RMSE train : {rmse_train:.3f}")
    print(f"RMSE eval  : {rmse_eval:.3f}")

    min_, max_ = np.quantile(X, (0, 1), axis=0)
    t1, t2 = np.linspace(min_, max_, 100).T
    N, M = np.meshgrid(t1, t2)
    N = N.ravel()
    M = M.ravel()
    X_plot = np.column_stack((N, M))
    y_preds_plot = (model.predict(X_plot) > 0.5).astype(int, copy=False)

    colors = list(map({0: "red", 1: "blue"}.get, y))
    colors_plot = list(map({0: (1, 0.8, 0.7), 1: (0.7, 0.8, 1.0)}.get, y_preds_plot))

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.scatter(*X_plot.T, c=colors_plot)
    ax.scatter(*X.T, color=colors)
    plt.show()


if __name__ == "__main__":
    _test()
