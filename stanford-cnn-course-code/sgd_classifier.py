"""Implement the Softmax Classifier.

The softmax classifier is a multiclass generalization of the
logist regression classifier.
"""
import typing as t

import numpy as np

VectorizedFuncType = t.Callable[[t.Union[np.ndarray, float]], float]


class SGDClassifier:
    """."""

    def __init__(self, func_loss: VectorizedFuncType,
                 func_loss_deriv: VectorizedFuncType):
        """."""
        self._num_classes = -1
        self._num_inst = -1
        self._num_attr = -1
        self._func_loss = func_loss
        self._func_grad = None  # type: VectorizedFuncType
        self._func_reg = None  # type: VectorizedFuncType

        self.weights = np.array([])  # type: np.ndarray
        self.learning_rate = -1.0
        self.reg_rate = -1.0
        self.batch_size = -1
        self.max_it = -1
        self.epsilon = -1.0

        self._valid_reg = {"l1", "l2", None}

    @classmethod
    def _add_bias(cls, X: np.ndarray) -> np.ndarray:
        """Concatenate a full column of 1's as the last ``X`` column."""
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    def _optimize(self, X: np.ndarray, y: np.ndarray, verbose: int) -> None:
        """."""
        cur_it = 0
        err_cur = 1 + self.epsilon
        err_prev = err_cur

        while cur_it < self.max_it and err_cur > self.epsilon:
            cur_it += 1

            sample_inds = np.random.choice(
                y.size, size=self.batch_size, replace=False)

            X_sample = X[sample_inds, :]
            y_sample = y[sample_inds]

            loss_inst = self._func_loss(X=X_sample, W=self.weights, y=y_sample)
            loss_reg = self.reg_rate * self._func_reg(W=self.weights)

            loss_total = loss_inst + loss_reg

            grad = self._func_grad(
                X=X_sample, W=self.weights, y=y_sample, loss=loss_total)

            self.weights -= self.learning_rate * grad

            if verbose:
                print("Iteration: {} of {} - Current average loss: {:.6f} "
                      "(relative change of {:.2f}%)".format(
                          cur_it, self.max_it, err_cur,
                          100 * (err_cur - err_prev) / err_prev))

            err_prev = err_cur
            err_cur = loss_total / self.batch_size

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = 256,
            max_it: int = 1000,
            learning_rate: float = 0.0001,
            regularization: t.Optional[str] = "l2",
            epsilon: float = 1.0e-6,
            add_bias: bool = True,
            verbose: int = 0,
            random_state: t.Optional[int] = None) -> "SGDClassifier":
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

        if regularization not in self._valid_reg:
            raise ValueError("Invalid 'regularization' value ({}). Choose a "
                             "value between {}.".format(
                                 regularization, self._valid_reg))

        if learning_rate <= 0:
            raise ValueError("'learning_rate' must be positive (got {}.)"
                             .format(learning_rate))

        if epsilon < 0:
            raise ValueError("'epsilon' must be non-negative (got {}.)"
                             .format(epsilon))

        self._num_classes = np.unique(y).size
        self._num_inst, self._num_attr = X.shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_it = max_it

        if add_bias:
            X = self._add_bias(X)

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(self._num_classes, 1 + self._num_attr))

        self._optimize(X=X, y=y, verbose=verbose)

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
        super().__init__(self)

    @classmethod
    def logistic_fun(
            cls, x: t.Union[np.ndarray, float]) -> t.Union[np.ndarray, float]:
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
