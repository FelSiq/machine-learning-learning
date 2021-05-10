import functools

import numpy as np
import scipy.spatial


class LWLR:
    DISTANCE_FUNC = {
        "gaussian": lambda d_sqr, bandwidth: np.exp(-0.5 / bandwidth ** 2 * d_sqr),
        "tricube": lambda d_sqr: np.power(1.0 - np.power(d_sqr, 1.5), 3),
    }

    def __init__(
        self,
        bandwidth: float = 1.0,
        distance_func: str = "tricube",
        add_intercept: bool = True,
    ):
        assert float(bandwidth) > 0.0
        assert distance_func in {"gaussian", "tricube"}

        self.X = np.empty(0)
        self.y = np.empty(0)

        self.add_intercept = add_intercept

        self.distance_func = distance_func
        self._dist_func = self.DISTANCE_FUNC[distance_func]
        self._scale_dist = distance_func == "tricube"

        if distance_func == "gaussian":
            self._dist_func = functools.partial(self._dist_func, bandwidth=bandwidth)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        self.X = X
        self.y = y

        return self

    def predict(self, X):
        X = np.asfarray(X)

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        dist_sqr = scipy.spatial.distance.cdist(X, self.X, metric="sqeuclidean")

        if self._scale_dist:
            max_, min_ = np.quantile(X, (0, 1))
            dist_sqr = np.square((np.sqrt(dist_sqr) - min_) / (max_ - min_))

        weights = self._dist_func(dist_sqr)

        preds = np.empty(n, dtype=float)

        for i in np.arange(n):
            theta = (
                np.linalg.inv(self.X.T @ np.diag(weights[i, :]) @ self.X)
                @ self.X.T
                @ (weights[i] * self.y)
            )
            pred = X[i, :] @ theta
            preds[i] = pred

        return preds


def _test_01():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.preprocessing

    X, y = sklearn.datasets.load_diabetes(return_X_y=True)

    n_splits = 10
    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )
    scaler = sklearn.preprocessing.StandardScaler()

    def rmse(a, b):
        return sklearn.metrics.mean_squared_error(a, b, squared=False)

    model = LWLR(bandwidth=4)

    rmse_train = rmse_eval = 0.0

    for inds_train, inds_eval in splitter.split(X, y):
        X_train, X_eval = X[inds_train, :], X[inds_eval, :]
        y_train, y_eval = y[inds_train], y[inds_eval]

        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        model.fit(X_train, y_train)
        y_preds_train = model.predict(X_train)
        y_preds_eval = model.predict(X_eval)

        rmse_train += rmse(y_preds_train, y_train)
        rmse_eval += rmse(y_preds_eval, y_eval)

    rmse_train /= n_splits
    rmse_eval /= n_splits
    baseline = np.std(y)

    print(f"baseline   : {baseline:.3f}")
    print(f"RMSE train : {rmse_train:.3f}")
    print(f"RMSE eval  : {rmse_eval:.3f}")


def _test_02():
    import matplotlib.pyplot as plt
    import sklearn.model_selection
    import sklearn.metrics

    n = 75
    X = np.linspace(0, 1, n) + 0.05 * np.random.randn(n)
    y = 0.5 * X + 2 + 0.01 * np.random.randn(n)

    X = X.reshape(-1, 1)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=16
    )

    model = LWLR(bandwidth=1)
    model.fit(X_train, y_train)

    y_preds_train = model.predict(X_train)
    y_preds_eval = model.predict(X_eval)

    baseline = np.std(y)
    train_rmse = sklearn.metrics.mean_squared_error(
        y_preds_train, y_train, squared=False
    )
    eval_rmse = sklearn.metrics.mean_squared_error(y_preds_eval, y_eval, squared=False)

    print(f"RMSE train : {train_rmse:.3f}")
    print(f"RMSE eval  : {eval_rmse:.3f}")

    plt.scatter(X_train, y_train, label="train")
    plt.scatter(X_eval, y_eval, label="eval")
    plt.scatter(X_train, y_preds_train, label="train (preds)")
    plt.scatter(X_eval, y_preds_eval, label="eval (preds)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test_01()
    # _test_02()
