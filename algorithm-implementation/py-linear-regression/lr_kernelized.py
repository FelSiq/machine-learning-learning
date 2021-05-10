import functools

import numpy as np
import scipy.spatial


class KernelLinReg:
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
            update = self.learning_rate * (kernel_mat @ self.coeffs - y)
            self.coeffs -= update

        return self

    def predict(self, X):
        X = np.asfarray(X)

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        kernel_mat = self._calc_kernel_mat(X, self.X)
        preds = kernel_mat @ self.coeffs

        return preds


def _test():
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.kernel_ridge

    np.random.seed(16)

    n = 100
    X = np.random.randn(n, 2)
    y = np.sum(X * 1.5 + np.exp(-np.square(X - 1)) + 30, axis=1)

    n_splits = 10

    splitter = sklearn.model_selection.KFold(n_splits=n_splits)
    model = KernelLinReg(bandwidth=10)
    ref = sklearn.kernel_ridge.KernelRidge(alpha=0.001, kernel="rbf", gamma=0.1)

    rmse_train = rmse_eval = 0.0
    rmse_train_ref = rmse_eval_ref = 0.0

    def rmse(a, b):
        return sklearn.metrics.mean_squared_error(a, b, squared=False)

    for inds_train, inds_eval in splitter.split(X, y):
        X_train, X_eval = X[inds_train, :], X[inds_eval, :]
        y_train, y_eval = y[inds_train], y[inds_eval]

        model.fit(X_train, y_train)
        ref.fit(X_train, y_train)

        y_preds_train = model.predict(X_train)
        y_preds_eval = model.predict(X_eval)
        y_preds_train_ref = ref.predict(X_train)
        y_preds_eval_ref = ref.predict(X_eval)

        rmse_train += rmse(y_preds_train, y_train)
        rmse_eval += rmse(y_preds_eval, y_eval)
        rmse_train_ref += rmse(y_preds_train_ref, y_train)
        rmse_eval_ref += rmse(y_preds_eval_ref, y_eval)

    rmse_train /= n_splits
    rmse_eval /= n_splits
    rmse_train_ref /= n_splits
    rmse_eval_ref /= n_splits

    print(f"(mine) RMSE train : {rmse_train:.3f}")
    print(f"(mine) RMSE eval  : {rmse_eval:.3f}")
    print(f"(ref)  RMSE train : {rmse_train_ref:.3f}")
    print(f"(ref)  RMSE eval  : {rmse_eval_ref:.3f}")


if __name__ == "__main__":
    _test()
