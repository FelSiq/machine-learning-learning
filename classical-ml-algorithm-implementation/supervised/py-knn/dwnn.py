"""dWNN Regressor."""
import numpy as np
import scipy.spatial
import scipy.stats


class DWNN:
    def __init__(self, scale: float = 1.0, p: int = 2):
        assert float(scale) > 0
        assert p > 0
        self.p = p
        self._gaussian_dist = scipy.stats.norm(scale=float(scale))

    def fit(self, X: np.ndarray, y: np.ndarray, copy: bool = False):
        self.X = X
        self.y = y

        if copy:
            self.X = np.copy(self.X)
            self.y = np.copy(self.y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = scipy.spatial.distance.cdist(X, self.X, p=self.p)
        weights = self._gaussian_dist.pdf(dists)
        weights /= np.sum(weights, axis=1, keepdims=True)
        preds = np.dot(weights, self.y)
        return preds


def _test():
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.neighbors
    import sklearn.preprocessing
    import tqdm.auto

    n_splits = 10
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    splitter = sklearn.model_selection.KFold(n_splits=n_splits)

    eval_rmse = ref_eval_rmse = 0.0

    for train_inds, eval_inds in tqdm.auto.tqdm(
        splitter.split(X, y), total=n_splits, leave=False
    ):
        X_train, X_eval = X[train_inds, :], X[eval_inds, :]
        y_train, y_eval = y[train_inds], y[eval_inds]

        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        model = DWNN()
        model.fit(X_train, y_train)
        y_preds = model.predict(X_eval)
        eval_rmse += sklearn.metrics.mean_squared_error(y_preds, y_eval, squared=False)

        ref = sklearn.neighbors.KNeighborsRegressor()
        ref.fit(X_train, y_train)
        y_preds = ref.predict(X_eval)
        ref_eval_rmse += sklearn.metrics.mean_squared_error(
            y_preds, y_eval, squared=False
        )

    eval_rmse /= n_splits
    ref_eval_rmse /= n_splits

    print(f"Eval rmse: {eval_rmse:.4f}")
    print(f"Eval rmse: {ref_eval_rmse:.4f}")


if __name__ == "__main__":
    _test()
