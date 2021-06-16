"""Simple implementation of Naive Bayes' Classifier algorithm."""
import typing as t

import numpy as np


class NaiveBayes:
    """Simple Naive Bayes' Classifier model."""

    def __init__(self):
        self._logpriors = np.empty(0)
        self._classes = np.empty(0)
        self._cls_freqs = np.empty(0)
        self._cls_loglikelihoods = dict()
        self._ls = 0

    @staticmethod
    def _check_X(X: np.ndarray) -> None:
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    @staticmethod
    def _check_y(y: np.ndarray) -> None:
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def _calc_category_prob(self, cat_val: float, class_id: int) -> float:
        prob = (cat_val + self._ls) / (
            self._cls_freqs[class_id] + self._classes.size * self._ls
        )
        return prob

    def _calc_category_freqs_by_cls(self, X: np.ndarray, y: np.ndarray) -> None:
        self._cls_loglikelihoods = {cls: dict() for cls in self._classes}

        for inst, cls in zip(X, y):
            for attr_category in enumerate(inst):
                self._cls_loglikelihoods[cls].setdefault(attr_category, 0.0)
                self._cls_loglikelihoods[cls][attr_category] += 1.0

    def _calc_loglikelihoods(self) -> None:
        for i, cls in enumerate(self._classes):
            attr_category_freqs = self._cls_loglikelihoods[cls]
            for attr_category, freq in attr_category_freqs.items():
                _likelihood = self._calc_category_prob(freq, i)
                _loglikelihood = np.log(_likelihood)
                attr_category_freqs[attr_category] = _loglikelihood

    def _calc_unknown_logprobs(self, eps: float = 1e-9):
        self._unknown_logprobs = np.array(
            [self._calc_category_prob(0, i) for i in range(self._classes.size)]
        )
        np.log(eps + self._unknown_logprobs, out=self._unknown_logprobs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, laplace_smoothing: bool = True
    ) -> "NaiveBayes":
        self._check_X(X)
        self._check_y(y)
        assert X.shape[0] == y.size

        self._ls = int(bool(laplace_smoothing))

        self._classes, self._cls_freqs = np.unique(y, return_counts=True)
        self._logpriors = np.log(self._cls_freqs.astype(np.float32) / len(y))

        self._calc_unknown_logprobs()
        self._calc_category_freqs_by_cls(X, y)
        self._calc_loglikelihoods()

        return self

    def predict(self, X: np.ndarray) -> t.List[t.Any]:
        self._check_X(X)
        preds = [None] * X.shape[0]

        for i, inst in enumerate(X):
            posterior_logprobs = np.copy(self._logpriors)

            for k, cls in enumerate(self._classes):
                for attr_category in enumerate(inst):
                    posterior_logprobs[k] += self._cls_loglikelihoods[cls].get(
                        attr_category, self._unknown_logprobs[k]
                    )

            preds[i] = self._classes[np.argmax(posterior_logprobs)]

        return preds


def _test():
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.preprocessing
    import sklearn.datasets

    X, y = sklearn.datasets.load_iris(return_X_y=True)

    n_splits = 10
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    discretizer = sklearn.preprocessing.KBinsDiscretizer(n_bins=4, encode="ordinal")

    avg_acc = 0.0

    for train_inds, test_inds in splitter.split(X, y):
        X_train, X_test = X[train_inds, ...], X[test_inds, ...]
        y_train, y_test = y[train_inds], y[test_inds]

        X_train = discretizer.fit_transform(X_train).astype(np.int32)
        X_test = discretizer.transform(X_test).astype(np.int32)

        model = NaiveBayes()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        avg_acc += sklearn.metrics.accuracy_score(preds, y_test)

    avg_acc /= n_splits

    print("10-fold avg accuracy:", avg_acc)


if __name__ == "__main__":
    _test()
