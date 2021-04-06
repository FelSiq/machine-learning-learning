"""Simple implementation of Gaussian Naive Bayes' Classifier algorithm."""
import typing as t

import numpy as np
import scipy.stats


class GaussianNaiveBayes:
    """Simple Gaussian Naive Bayes' Classifier model."""

    def __init__(self):
        self._logpriors = np.empty(0)
        self._classes = np.empty(0)
        self._cls_freqs = np.empty(0)

        self._dists = []  # type: t.List[scipy.stats.multivariate_normal]

    @staticmethod
    def _check_X(X: np.ndarray) -> None:
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    @staticmethod
    def _check_y(y: np.ndarray) -> None:
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def _calc_gaussian_dists(self, X: np.ndarray, y: np.ndarray) -> None:
        self._dists = []

        for cls, cls_ind in enumerate(self._classes):
            X_slice = X[self._classes[cls_ind] == y, :]

            mean = np.mean(X_slice, axis=0)
            cov = np.var(X_slice, axis=0, ddof=1) + 1e-4

            dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            self._dists.append(dist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        self._check_X(X)
        self._check_y(y)
        assert X.shape[0] == y.size

        self._classes, self._cls_freqs = np.unique(y, return_counts=True)
        self._logpriors = np.log(self._cls_freqs.astype(np.float32) / len(y))
        self._calc_gaussian_dists(X, y)
        return self

    def predict(self, X: np.ndarray) -> t.List[t.Any]:
        self._check_X(X)

        preds = np.full(X.shape[0], fill_value=-1, dtype=int)
        best_posterior_logprobs = np.full(len(preds), fill_value=-np.inf)

        for cls, cls_ind in enumerate(self._classes):
            likelihoods = self._dists[cls_ind].logpdf(X)
            cls_posterior_logprobs = likelihoods + self._logpriors[cls_ind]

            update_inds = best_posterior_logprobs < cls_posterior_logprobs

            preds[update_inds] = cls_ind
            best_posterior_logprobs[update_inds] = cls_posterior_logprobs[update_inds]

        preds = self._classes[preds]

        return preds


def _test():
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.preprocessing
    import sklearn.datasets

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    n_splits = 10
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    avg_acc = 0.0

    for train_inds, test_inds in splitter.split(X, y):
        X_train, X_test = X[train_inds, ...], X[test_inds, ...]
        y_train, y_test = y[train_inds], y[test_inds]

        model = GaussianNaiveBayes()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        avg_acc += sklearn.metrics.accuracy_score(preds, y_test)

    avg_acc /= n_splits
    print("10-fold avg accuracy:", avg_acc)


if __name__ == "__main__":
    _test()
