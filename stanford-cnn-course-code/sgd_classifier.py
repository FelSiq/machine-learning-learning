"""Implement the Softmax Classifier.

The softmax classifier is a multiclass generalization of the
logist regression classifier.
"""
import typing as t

import numpy as np


class SGDClassifier:
    """."""

    def __init__(
            self,
            loss_func: t.Callable[[t.Union[np.ndarray, float]], float],
            loss_func_deriv: t.Callable[[t.Union[np.ndarray, float]], float]):
        """."""
        self.weights = None
        self._num_classes = -1
        self._num_inst = -1
        self._num_attr = -1

    @classmethod
    def _add_bias(cls, X: np.ndarray) -> np.ndarray:
        """Concatenate a full column of 1's as the last ``X`` column."""
        return np.hstack((X, np.ones(X.shape[0], 1, dtype=X.dtype)))

    def _optimize(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
        """."""

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = 256,
            max_it: int = 1000,
            add_bias: bool = True,
            random_state: t.Optional[int] = None) -> "SoftmaxClassifier":
        """."""
        if not 0 < batch_size <= y.size:
            raise ValueError("'batch_size' must be a positive integer "
                             "smaller than the number of instances (got "
                             "{}.)".format(batch_size))

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.size:
            raise ValueError("'X' ({} instances) and 'y' ({} instances) "
                             "dimensions does not match!".format(
                                 X.shape[0], y.size))

        self._num_classes = np.unique(y).size
        self._num_inst, self._num_attr = X.shape

        if add_bias:
            X = self._add_bias(X)

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(self._num_classes, 1 + self._num_attr))

        self._optimize(X=X, y=y, batch_size=batch_size)

        return self

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """Calculate the linear scores based on fitted weights."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if add_bias:
            X = self._add_bias(X)

        return np.dot(self.weights, X.T)


class SoftmaxClassifier(SGDClassifier):
    """."""

    def __init__(self):
        """."""
        super.__init__()

    @classmethod
    def logistic_fun(
            x: t.Union[np.ndarray, float]) -> t.Union[np.ndarray, float]:
        """."""
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def logistic_fun_deriv(
            cls, x: t.Union[np.ndarray, float]) -> t.Union[np.ndarray, float]:
        """."""
        _aux = cls.logistic_fun(x)
        return (1.0 - _aux) * _aux


def _test() -> None:
    """."""


if __name__ == "__main__":
    _test()
