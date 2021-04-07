"""k-NN Classifier."""
import numpy as np
import scipy.spatial
import scipy.stats
import tqdm.auto


class _KNNBase:
    def __init__(self, k: int = 3, metric: str = "minkowski", p: int = 2):
        assert int(k) > 0
        assert p > 0

        self.k = int(k)
        self.p = p
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray, copy: bool = False):
        self.X = X
        self.y = y

        if copy:
            self.X = np.copy(self.X)
            self.y = np.copy(self.y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = scipy.spatial.distance.cdist(X, self.X, p=self.p, metric=self.metric)
        knn_inds = np.argsort(dists, axis=1)[:, : self.k]
        knn_labels = self.y[knn_inds]
        return self._prepare_output(knn_labels)


class KNNClassifier(_KNNBase):
    @classmethod
    def _prepare_output(cls, knn_labels: np.ndarray) -> np.ndarray:
        preds, _ = scipy.stats.mode(knn_labels, axis=1)
        return preds[:, 0]


class KNNRegressor(_KNNBase):
    @classmethod
    def _prepare_output(cls, knn_labels: np.ndarray) -> np.ndarray:
        preds = np.mean(knn_labels, axis=1)
        return preds


def _test():
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.neighbors
    import sklearn.preprocessing

    classifier = True
    n_splits = 10
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    splitter = sklearn.model_selection.KFold(n_splits=n_splits)
    k = 3

    eval_perf = ref_eval_perf = 0.0

    for train_inds, eval_inds in tqdm.auto.tqdm(
        splitter.split(X, y), total=n_splits, leave=False
    ):
        X_train, X_eval = X[train_inds, :], X[eval_inds, :]
        y_train, y_eval = y[train_inds], y[eval_inds]

        scaler = sklearn.preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        if classifier:
            model = KNNClassifier(k=k)
            ref = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)

        else:
            model = KNNRegressor(k=k)
            ref = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)

        model.fit(X_train, y_train)
        ref.fit(X_train, y_train)

        y_preds = model.predict(X_eval)
        y_preds = ref.predict(X_eval)

        if classifier:
            eval_perf += sklearn.metrics.accuracy_score(y_preds, y_eval)
            ref_eval_perf += sklearn.metrics.accuracy_score(y_preds, y_eval)

        else:
            eval_perf += sklearn.metrics.mean_squared_error(y_preds, y_eval, squared=False)
            ref_eval_perf += sklearn.metrics.mean_squared_error(y_preds, y_eval, squared=False)

    eval_perf /= n_splits
    ref_eval_perf /= n_splits

    if classifier:
        print(f"Eval acc: {eval_perf:.4f}")
        print(f"Eval acc: {ref_eval_perf:.4f}")
    else:
        print(f"Eval rmse: {eval_perf:.4f}")
        print(f"Eval rmse: {ref_eval_perf:.4f}")


if __name__ == "__main__":
    _test()
