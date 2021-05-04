import typing as t

import numpy as np
import scipy.stats


class GP:
    def __init__(
        self,
        kernel: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
        noise_var: t.Optional[float] = None,
    ):
        assert noise_var is None or float(noise_var) > 0.0
        self.kernel = kernel
        self.X = np.empty(0)
        self.noise_var = noise_var

    def fit(self, X, y):
        self.X = np.array(X, dtype=float)
        y = np.asfarray(y)

        if self.noise_var is None:
            self.noise_var = np.var(y, ddof=1)

        n, m = X.shape

        kernel_mat = self.kernel(X, X)

        eye_var = np.diag(np.full(n, fill_value=self.noise_var))
        self.aux_inv = np.linalg.inv(kernel_mat + eye_var)
        self.aux_mu = self.aux_inv @ y

        return self

    def predict(self, X):
        n, m = X.shape

        kernel_mat = self.kernel(X, self.X)
        kernel_mat_pred = self.kernel(X, X)

        mu = kernel_mat @ self.aux_mu

        eye_var = np.diag(np.full(n, fill_value=self.noise_var))
        sigma = kernel_mat_pred + eye_var - kernel_mat @ self.aux_inv @ kernel_mat.T

        return mu


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.gaussian_process

    X, y = sklearn.datasets.load_boston(return_X_y=True)
    n_splits = 10
    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    rmse_train = rmse_eval = 0.0
    rmse_train_ref = rmse_eval_ref = 0.0

    kernel = sklearn.gaussian_process.kernels.RBF()
    model = GP(kernel=kernel)
    ref = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)

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

    baseline = np.var(y)

    print(f"Baseline          : {baseline:.3f}")
    print(f"(mine) Train rmse : {rmse_train:.3f}")
    print(f"(mine) Eval rmse  : {rmse_eval:.3f}")
    print(f"(ref)  Train rmse : {rmse_train_ref:.3f}")
    print(f"(ref)  Eval rmse  : {rmse_eval_ref:.3f}")


if __name__ == "__main__":
    _test()
