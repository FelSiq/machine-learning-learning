import typing as t

import numpy as np
import scipy.stats


class GP:
    def __init__(self, kernel: t.Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.kernel = kernel
        self.X_train = np.empty(0)

        self.aux_inv = np.empty(0)
        self.aux_mu = np.empty(0)

        self.y_mean = 0.0
        self.y_std = 0.0

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        y = np.asfarray(y).ravel()

        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

        n, m = X.shape

        kernel_mat = self.kernel(X, X)

        self.aux_inv = np.linalg.inv(kernel_mat + np.eye(n))
        self.aux_mu = self.aux_inv @ (y - self.y_mean)

        return self

    def predict(self, X, return_std: bool = False, return_cov: bool = False):
        n, m = X.shape

        kernel_mat = self.kernel(X, self.X_train)
        kernel_mat_pred = self.kernel(X, X)

        mu = kernel_mat @ self.aux_mu + self.y_mean

        if return_std or return_cov:
            sigma = (
                kernel_mat_pred + np.eye(n) - kernel_mat @ self.aux_inv @ kernel_mat.T
            ) * np.square(self.y_std)

        ret = [mu]

        if return_std:
            std = np.sqrt(np.diag(sigma))
            ret.append(std)

        if return_cov:
            ret.append(sigma)

        return tuple(ret) if len(ret) > 1 else ret[0]


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.gaussian_process

    X, y = sklearn.datasets.load_boston(return_X_y=True)

    n_splits = 5
    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    rmse_train = rmse_eval = 0.0
    rmse_train_ref = rmse_eval_ref = 0.0

    kernel = (
        sklearn.gaussian_process.kernels.WhiteKernel()
        + sklearn.gaussian_process.kernels.DotProduct()
    )
    model = GP(kernel=kernel)
    ref = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=kernel, normalize_y=True
    )

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

    baseline = np.std(y)

    print(f"Baseline          : {baseline:.3f}")
    print(f"(mine) Train rmse : {rmse_train:.3f}")
    print(f"(mine) Eval rmse  : {rmse_eval:.3f}")
    print(f"(ref)  Train rmse : {rmse_train_ref:.3f}")
    print(f"(ref)  Eval rmse  : {rmse_eval_ref:.3f}")


if __name__ == "__main__":
    _test()
