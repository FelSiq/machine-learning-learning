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

            means = np.mean(X_slice, axis=0)
            stds = np.std(X_slice, axis=0)

            # Note: we could actually implement this using a single
            # multivariate Gaussian per class, constraining it to have
            # a diagonal covariance matrix. Although this would seem to
            # be a more concise and clear implementation, that would
            # take O(m ** 2) space instead of O(m), where m is the number
            # of features in X, so I opt for 'm' distinct univariate
            # Gaussians instead.
            dist = tuple(scipy.stats.norm(loc=m, scale=s) for m, s in zip(means, stds))
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
            likelihoods = np.sum(
                [dist.logpdf(X[:, i]) for i, dist in enumerate(self._dists[cls_ind])],
                axis=0,
            )
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
    import sklearn.naive_bayes

    X, y = sklearn.datasets.load_wine(return_X_y=True)

    n_splits = 10
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    avg_acc = avg_acc_ref = 0.0

    model = GaussianNaiveBayes()
    ref = sklearn.naive_bayes.GaussianNB()

    for train_inds, eval_inds in splitter.split(X, y):
        X_train, X_eval = X[train_inds, ...], X[eval_inds, ...]
        y_train, y_eval = y[train_inds], y[eval_inds]

        model.fit(X_train, y_train)
        ref.fit(X_train, y_train)

        y_preds = model.predict(X_eval)
        y_preds_ref = ref.predict(X_eval)

        avg_acc += sklearn.metrics.accuracy_score(y_eval, y_preds)
        avg_acc_ref += sklearn.metrics.accuracy_score(y_eval, y_preds_ref)

    avg_acc /= n_splits
    avg_acc_ref /= n_splits

    print(f"(mine)    10-fold avg accuracy: {avg_acc:.3f}")
    print(f"(sklearn) 10-fold avg accuracy: {avg_acc_ref:.3f}")


if __name__ == "__main__":
    _test()
