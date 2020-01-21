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

    def __init__(self, func_loss: t.Callable, func_loss_grad: t.Callable):
        """."""
        self._num_classes = -1
        self._num_inst = -1
        self._num_attr = -1
        self._func_loss = func_loss
        self._func_loss_grad = func_loss_grad
        self._func_reg = None  # type: t.Optional[VectorizedFuncType]

        self.weights = np.array([])  # type: np.ndarray
        self.learning_rate = -1.0
        self.reg_rate = -1.0
        self.batch_size = -1
        self.max_it = -1
        self.epsilon = -1.0

        self.errors = None  # type: t.Optional[np.ndarray]

    @classmethod
    def _add_bias(cls, X: np.ndarray) -> np.ndarray:
        """Concatenate a full column of 1's as the last ``X`` column."""
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    def _optimize(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  verbose: int,
                  store_errors: bool = False) -> None:
        """."""
        cur_it = 0
        err_cur = 1 + self.epsilon
        err_prev = err_cur

        if store_errors:
            self.errors = np.zeros(self.max_it, dtype=float)

        while cur_it < self.max_it and err_cur > self.epsilon:

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

            err_prev = err_cur
            err_cur = loss_total / self.batch_size

            if store_errors:
                self.errors[cur_it] = err_cur

            cur_it += 1

            if verbose:
                print("Iteration: {} of {} - Current average loss: {:.6f} "
                      "(relative change of {:.2f}%)".format(
                          cur_it, self.max_it, err_cur,
                          100 * (err_cur - err_prev) / err_prev))

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = 256,
            max_it: int = 5000,
            learning_rate: float = 0.0001,
            reg_rate: float = 0.0001,
            epsilon: float = 1.0e-6,
            add_bias: bool = True,
            store_errors: bool = False,
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

        self._optimize(X=X, y=y, verbose=verbose, store_errors=store_errors)

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

    def cross_ent_grad(self,
                       X: np.ndarray,
                       y_inds: np.ndarray,
                       scores: t.Optional[np.ndarray] = None,
                       add_bias: bool = True) -> np.ndarray:
        """."""
        if scores is None:
            scores = super()._predict(X=X, add_bias=add_bias)

        _scores = self.softmax(scores=scores, axis=0)

        correct_class_ind = np.zeros((self._num_classes, y_inds.size))
        correct_class_ind[y_inds, np.arange(y_inds.size)] = 1

        loss_grad_reg = 2 * self.reg_rate * self.weights
        loss_grad_score = np.dot(_scores - correct_class_ind, X)
        loss_total = loss_grad_score / y_inds.size + loss_grad_reg

        return loss_total


class SupportVectorClassifier(SGDClassifier):
    """."""

    def __init__(self):
        """."""
        super().__init__(
            func_loss=losses.hinge_loss, func_loss_grad=self.hinge_loss_grad)

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """."""
        return np.argmax(super()._predict(X=X, add_bias=add_bias), axis=0)

    def hinge_loss_grad(self, X: np.ndarray, y_inds: np.ndarray,
                        scores: np.ndarray) -> np.ndarray:
        """."""
        loss_grad_reg = 2 * self.reg_rate * self.weights
        loss_grad_score = None  # TODO
        loss_total = loss_grad_score / y_inds.size + loss_grad_reg

        return loss_total


def _gen_data(inst_per_class: int = 200) -> t.Tuple[np.ndarray, np.ndarray]:
    """Generate multimodal data."""
    X = 0.85 * np.vstack((
        np.random.multivariate_normal(
            mean=(2, 2), cov=np.eye(2), size=inst_per_class),
        np.random.multivariate_normal(
            mean=(-3, -3), cov=np.eye(2), size=inst_per_class),
        np.random.multivariate_normal(
            mean=(0, 6), cov=np.eye(2), size=inst_per_class),
    )) + 0.15 * np.random.multivariate_normal(
        mean=(0, 0), cov=np.eye(2), size=3 * inst_per_class)

    y = np.repeat(np.arange(3), inst_per_class).astype(int)

    return X, y


def _test_softmax_classifier() -> None:
    """Full experiment with the implemented SoftmaxClassifier."""
    import matplotlib.pyplot as plt
    import sklearn.model_selection

    model = SoftmaxClassifier()

    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=32)

    folds = sklearn.model_selection.StratifiedKFold(n_splits=5)

    learning_rate_candidates = np.logspace(-0.5, 0.2, 10)
    learning_rate_acc = np.zeros(learning_rate_candidates.size)

    for l_ind, learning_rate in enumerate(learning_rate_candidates):
        print("Checking {} out of {} learning rates: {}...".format(
            l_ind, learning_rate_candidates.size, learning_rate))
        for ind, (inds_train, inds_test) in enumerate(
                folds.split(X_train, y_train)):
            model.fit(
                X_train[inds_train, :],
                y_train[inds_train],
                batch_size=32,
                max_it=2000,
                verbose=0,
                learning_rate=learning_rate)

            preds = model.predict(X_train[inds_test, :])
            learning_rate_acc[ind] += np.sum(
                preds == y_train[inds_test]) / inds_test.size

    best_learning_rate = learning_rate_candidates[np.argmax(learning_rate_acc)]
    print("Best learning rate:", best_learning_rate)

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        verbose=1,
        store_errors=True,
        learning_rate=best_learning_rate)

    preds = model.predict(X_test)
    print("Accuracy:", np.sum(preds == y_test) / y_test.size)

    plt.suptitle("Step size: {:.8f}".format(best_learning_rate))

    plt.subplot(1, 2, 1)
    plt.title("Instances Scatter plot")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker="X")

    plt.subplot(1, 2, 2)
    plt.title("Errors per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy Error")
    plt.plot(model.errors)

    plt.show()


def _test_softmax_grad() -> None:
    import gradient_check

    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    X = np.hstack((X, np.ones((y.size, 1))))

    reg_rate = 0.0001

    func = lambda W: losses.cross_ent_loss(X=X,
                   y_inds=y,
                   W=W.reshape((3, 2 + 1)),
                   lambda_=reg_rate)

    def func_grad(W: np.ndarray):
        model = SoftmaxClassifier()
        model.fit(X, y, max_it=0)
        model.weights = W.reshape((3, 2 + 1))
        return model.cross_ent_grad(X=X, y_inds=y, add_bias=False).ravel()

    error = gradient_check.gradient_check(
        func=func,
        analytic_grad=func_grad,
        x_limits=np.array([-5, 5] * 9).reshape(-1, 2),
        random_state=32,
        verbose=1)

    print("Gradient check error:", error)


if __name__ == "__main__":
    _test_softmax_grad()
    # _test_softmax_classifier()
    # _test_support_vector_classifier()
