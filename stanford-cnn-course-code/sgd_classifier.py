"""Implement the Softmax Classifier.

The softmax classifier is a multiclass generalization of the
logist regression classifier.
"""
import typing as t

import numpy as np

import losses

VectorizedFuncType = t.Callable[[t.Union[np.ndarray, float]], float]


class SGDClassifier:
    """."""

    def __init__(
            self, func_loss: VectorizedFuncType,
            func_loss_grad: t.Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """."""
        self._num_classes = -1
        self._num_inst = -1
        self._num_attr = -1
        self._func_loss = func_loss
        self._func_loss_grad = func_loss_grad
        self._func_reg = None  # type: VectorizedFuncType

        self.weights = np.array([])  # type: np.ndarray
        self.learning_rate = -1.0
        self.reg_rate = -1.0
        self.batch_size = -1
        self.max_it = -1
        self.epsilon = -1.0

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

            scores = self._predict(X=X_sample, add_bias=False)

            loss_total = self._func_loss(
                X=X_sample,
                y_inds=y_sample,
                W=self.weights,
                scores=scores,
                lambda_=self.reg_rate)

            grad = self._func_loss_grad(
                X=X_sample, y_inds=y_sample, scores=scores)

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
            reg_rate: float = 0.0001,
            epsilon: float = 1.0e-6,
            add_bias: bool = True,
            verbose: int = 0,
            random_state: t.Optional[int] = None) -> "SGDClassifier":
        """."""
        if not 0 < batch_size <= y.size:
            batch_size = y.size

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.size:
            raise ValueError("'X' ({} instances) and 'y' ({} instances) "
                             "dimensions does not match!".format(
                                 X.shape[0], y.size))

        if learning_rate <= 0:
            raise ValueError("'learning_rate' must be positive (got {}.)".
                             format(learning_rate))

        if epsilon < 0:
            raise ValueError(
                "'epsilon' must be non-negative (got {}.)".format(epsilon))

        self._num_classes = np.unique(y).size
        self._num_inst, self._num_attr = X.shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_it = max_it
        self.reg_rate = reg_rate

        if add_bias:
            X = self._add_bias(X)

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(self._num_classes, 1 + self._num_attr))

        self._optimize(X=X, y=y, verbose=verbose)

        return self

    def _predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """Calculate the linear scores based on fitted weights."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if add_bias:
            X = self._add_bias(X)

        scores = np.dot(self.weights, X.T)

        return scores

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """Calculate the linear scores based on fitted weights."""
        return self._predict(X=X, add_bias=add_bias)


class SoftmaxClassifier(SGDClassifier):
    """."""

    def __init__(self):
        """."""
        super().__init__(
            func_loss=losses.cross_ent_loss,
            func_loss_grad=self.cross_ent_grad)

    @classmethod
    def softmax(cls, scores: np.ndarray,
                axis: t.Optional[int] = None) -> np.ndarray:
        """Compute the Softmax function."""
        # Note: subtract the maximum for numeric stability
        _scores_exp = np.exp(scores - np.max(scores, axis=axis))
        return _scores_exp / np.sum(_scores_exp, axis=axis)

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """."""
        scores = super()._predict(X=X, add_bias=add_bias)
        scores_norm = self.softmax(scores, axis=0)
        return np.argmax(scores_norm, axis=0)

    def cross_ent_grad(self, X: np.ndarray, y_inds: np.ndarray,
                       scores: np.ndarray) -> np.ndarray:
        """."""
        _scores = self.softmax(scores=scores, axis=0)

        correct_class_ind = np.zeros((self._num_classes, y_inds.size))
        correct_class_ind[y_inds, np.arange(y_inds.size)] = 1

        loss_grad_reg = 2 * self.reg_rate * self.weights
        loss_grad_score = np.dot(_scores - correct_class_ind, X)
        loss_total = loss_grad_score / y_inds.size + loss_grad_reg

        return loss_total


def _test() -> None:
    """."""
    import matplotlib.pyplot as plt
    import sklearn.model_selection

    model = SoftmaxClassifier()

    inst_per_class = 200

    np.random.seed(16)

    X = np.vstack((
        np.random.multivariate_normal(
            mean=(2, 2), cov=np.eye(2), size=inst_per_class),
        np.random.multivariate_normal(
            mean=(-3, -3), cov=np.eye(2), size=inst_per_class),
        np.random.multivariate_normal(
            mean=(0, 6), cov=np.eye(2), size=inst_per_class),
    ))

    y = np.repeat(np.arange(3), inst_per_class).astype(int)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=32)

    model.fit(X_train, y_train, batch_size=20)
    preds = model.predict(X_test)
    print("Accuracy:", np.sum(preds == y_test) / y_test.size)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker="X")
    plt.show()


if __name__ == "__main__":
    _test()
