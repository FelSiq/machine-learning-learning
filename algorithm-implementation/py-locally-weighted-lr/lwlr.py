import numpy as np
import scipy.spatial


class LWLR:
    def __init__(self, bandwidth: float = 1.0, add_intercept: bool = True):
        assert float(bandwidth) > 0.0

        self.X = np.empty(0)
        self.y = np.empty(0)

        self._norm_const = -0.5 / float(bandwidth) ** 2

        self.add_intercept = add_intercept

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

        self.mat = np.linalg.inv(X.T @ X) @ X.T * self.y

        return self

    def predict(self, X):
        X = np.asfarray(X)

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        dist_sqr = scipy.spatial.distance.cdist(X, self.X, metric="sqeuclidean")
        weights = np.exp(self._norm_const * dist_sqr)

        preds = np.sum(X @ self.mat * weights, axis=1)

        return preds


def _test():
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

    model = LWLR(bandwidth=10)

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


if __name__ == "__main__":
    _test()
