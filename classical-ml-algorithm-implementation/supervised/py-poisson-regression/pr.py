import numpy as np


class PoissonRegressor:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_epochs: int = 128,
        batch_size: int = 32,
        add_intercept: bool = True,
    ):
        assert float(learning_rate) > 0.0
        assert int(max_epochs) > 0
        assert int(batch_size) > 0

        self.coeffs = np.empty(0)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.add_intercept = add_intercept
        self.batch_size = int(batch_size)

        self._training = False

    def _run_epoch(self, X, y, batch_inds):
        for start in np.arange(0, y.size, self.batch_size):
            end = start + self.batch_size

            X_batch = X[batch_inds[start:end], :]
            y_batch = y[batch_inds[start:end]]

            y_preds = self.predict(X_batch)
            grad = X_batch.T @ (y_preds - y_batch)

            update = self.learning_rate * grad / self.batch_size
            self.coeffs -= update

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asfarray(y)

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, m = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        self.coeffs = np.random.randn(m + self.add_intercept)

        self._training = True
        batch_inds = np.arange(n)

        for i in np.arange(1, 1 + self.max_epochs):
            np.random.shuffle(batch_inds)
            self._run_epoch(X, y, batch_inds)

        self._training = False

        return self

    def predict(self, X):
        X = np.asfarray(X)
        n, _ = X.shape

        if not self._training and self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        preds = np.exp(X @ self.coeffs)

        return preds


def _test():
    import sklearn.model_selection
    import sklearn.linear_model
    import sklearn.metrics

    np.random.seed(16)
    X = np.random.randn(150, 3)
    true_coeffs = np.random.randn(3)
    y = np.exp(X @ true_coeffs + 0.25) + 0.5 * np.random.random(X.shape[0])

    def rmse(a, b):
        return sklearn.metrics.mean_squared_error(a, b, squared=False)

    model = PoissonRegressor(learning_rate=1e-1, max_epochs=512)
    ref = sklearn.linear_model.PoissonRegressor(alpha=1e-1, max_iter=512)

    n_splits = 10
    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    rmse_train = rmse_eval = 0.0
    rmse_train_ref = rmse_eval_ref = 0.0

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

    baseline = np.std(y)
    rmse_train /= n_splits
    rmse_eval /= n_splits
    rmse_train_ref /= n_splits
    rmse_eval_ref /= n_splits

    print(f"baseline          : {baseline:.3f}")
    print(f"(mine) RMSE train : {rmse_train:.3f}")
    print(f"(mine) RMSE eval  : {rmse_eval:.3f}")
    print(f"(ref)  RMSE train : {rmse_train_ref:.3f}")
    print(f"(ref)  RMSE eval  : {rmse_eval_ref:.3f}")

    print("True coeffs      :", np.hstack((0.25, true_coeffs)))
    print("Estimated coeffs :", model.coeffs)


if __name__ == "__main__":
    _test()
