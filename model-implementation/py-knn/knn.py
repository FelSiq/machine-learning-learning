"""k-NN Classifier."""
import numpy as np
import scipy.spatial
import scipy.stats
import tqdm.auto


class KNN:
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
        preds, _ = scipy.stats.mode(self.y[knn_inds], axis=1)
        return preds[:, 0]


def _test():
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.neighbors
    import sklearn.preprocessing

    n_splits = 10
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    splitter = sklearn.model_selection.KFold(n_splits=n_splits)
    k = 3

    eval_acc = ref_eval_acc = 0.0

    for train_inds, eval_inds in tqdm.auto.tqdm(
        splitter.split(X, y), total=n_splits, leave=False
    ):
        X_train, X_eval = X[train_inds, :], X[eval_inds, :]
        y_train, y_eval = y[train_inds], y[eval_inds]

        scaler = sklearn.preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_eval)
        eval_acc += sklearn.metrics.accuracy_score(y_preds, y_eval)

        ref = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        ref.fit(X_train, y_train)
        y_preds = ref.predict(X_eval)
        ref_eval_acc += sklearn.metrics.accuracy_score(y_preds, y_eval)

    eval_acc /= n_splits
    ref_eval_acc /= n_splits

    print(f"Eval acc: {eval_acc:.4f}")
    print(f"Eval acc: {ref_eval_acc:.4f}")


if __name__ == "__main__":
    _test()
