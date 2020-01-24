"""Implement Stochastic Gradient Descent Linear Classifiers."""
import typing as t

import numpy as np
import sklearn.model_selection

import losses
import regularization

VectorizedFuncType = t.Callable[[t.Union[np.ndarray, float]], float]


class SGDClassifier:
    """Implement Stochastic Gradient Descent Linear Classifiers.

    Actually, it is implemented a more general concept of optimization
    using Mini-Batches. When the Mini-batch has size of 1, then it is
    the special case of Stochastic Gradient Descent. Nevertheless,
    while incorrectly used, the technical term `SGD` os much more widely
    known, and hence this module name.

    The linear model here has the form f(X, W) = W * X^{T}, while W is
    the weights that must be optimized using Gradient Descent.

    `Gradient Descent` is just the algorithm of iteratively calculating
    the gradient for each current `W` value, and making small steps in
    the opposite direction of the gradient (where the function decrease
    the fastest considering an infinitesimal-sized step). Both algorithm
    steps are repeated many times, until some convergence criterion is
    meet.
    """

    def __init__(self,
                 func_loss: t.Callable,
                 func_loss_grad: t.Callable,
                 check_data_func: t.Optional[
                     t.Callable[[np.ndarray, np.ndarray], None]] = None):
        """Init SGD model.

        Arguments
        ---------
        func_loss : :obj:`callable`
            Loss function, also known as the objective function. This is
            and extremely important parameter, as it essentially has the
            power to define which algorithm is used to fit the model.
            For instance, if ``func_loss`` is the `cross entropy loss,`
            then the algorithm used is the `Softmax classification`,
            and if ``func_loss`` is the `hinge loss`, then the algorithm
            used is the Support Vector Classifier (the Support Vector
            Machine with Linear Kernel.)

        func_loss_grad : :obj:`callable`
            Gradient function for the chosen ``func_loss`` function.

        check_data_func : :obj:`callable`, optional
            Function used to verify if the fitted data follows the
            algorithm standards.
        """
        self._func_loss = func_loss
        self._func_reg = regularization.l2
        self._func_loss_grad = func_loss_grad
        self._func_reg_grad = regularization.l2_grad
        self._check_data_func = check_data_func

        self._num_classes = -1
        self._num_inst = -1
        self._num_attr = -1
        self._data_mean = np.array([])

        self.weights = np.array([])  # type: np.ndarray
        self.learning_rate = -1.0
        self.reg_rate = -1.0
        self.batch_size = -1
        self.max_epochs = -1
        self.epsilon = -1.0
        self.patience = -1
        self.patience_margin = 0.001
        self.epochs = -1

        self.errors = np.array([])

    @classmethod
    def _add_bias(cls, X: np.ndarray) -> np.ndarray:
        """Concatenate a full column of 1's as the last ``X`` column."""
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    def _calc_loss_total(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         scores: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate the total loss (data + regularization loss) of data."""
        if scores is None:
            scores = self._predict(X=X, center_data=False, add_bias=False)

        loss_reg = self._func_reg(
            W=self.weights, lambda_=self.reg_rate, exclude_bias=True)

        loss_data = self._func_loss(
            X=X, y_inds=y, W=self.weights, scores=scores)

        return loss_data + loss_reg

    def _calc_grad_total(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         scores: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate the total gradient (data + regularization) on data."""
        if scores is None:
            scores = self._predict(X=X, center_data=False, add_bias=False)

        grad_loss_reg = self._func_reg_grad(
            W=self.weights, lambda_=self.reg_rate, exclude_bias=True)

        grad_loss_data = self._func_loss_grad(
            X=X, y_inds=y, scores=scores, add_bias=False)

        return grad_loss_data + grad_loss_reg

    def _optimize(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_val: t.Optional[np.ndarray] = None,
                  y_val: t.Optional[np.ndarray] = None,
                  verbose: int = 0,
                  recover_best_weight: bool = False,
                  store_errors: bool = False) -> None:
        """Optimize the weights using SGD strategy."""
        epoch_inst_count = 0
        patience_ticks = 0

        err_train_cumulative = 0.0

        err_val_cumulative = 0.0
        err_val_epoch_mean = self._calc_loss_total(X=X_val, y=y_val)
        err_val_prev = err_val_epoch_mean
        err_val_best = err_val_epoch_mean

        if recover_best_weight:
            best_weights = np.copy(self.weights)

        if store_errors:
            self.errors = np.zeros((2, self.max_epochs), dtype=float)

        while (self.epochs < self.max_epochs
               and err_val_epoch_mean > self.epsilon
               and patience_ticks < self.patience):
            sample_inds = np.random.choice(
                y_train.size, size=self.batch_size, replace=False)

            X_sample = X_train[sample_inds, :]
            y_sample = y_train[sample_inds]

            scores = self._predict(
                X=X_sample, center_data=False, add_bias=False)

            loss_total = self._calc_loss_total(
                X=X_sample, y=y_sample, scores=scores)
            grad_total = self._calc_grad_total(
                X=X_sample, y=y_sample, scores=scores)

            self.weights -= self.learning_rate * grad_total

            err_train_cumulative += loss_total
            err_val_cumulative += self._calc_loss_total(X=X_val, y=y_val)

            epoch_inst_count += self.batch_size

            if epoch_inst_count >= self._num_inst:
                err_train_epoch_mean = err_train_cumulative / self.batch_size
                err_val_epoch_mean = err_val_cumulative / self.batch_size

                if store_errors:
                    self.errors[:, self.
                                epochs] = err_train_epoch_mean, err_val_epoch_mean

                if err_val_epoch_mean >= (
                        1.0 - self.patience_margin) * err_val_best:
                    patience_ticks += 1

                else:
                    err_val_best = err_val_epoch_mean
                    patience_ticks = 0

                    if recover_best_weight:
                        best_weights = np.copy(self.weights)

                if verbose:
                    print(
                        "Epoch: {} out of {} - Avg. epoch loss: [train] {:.6f} "
                        "[val.] {:.6f} (best: {:.6f}, rel. change of {:.2f}%)".
                        format(
                            self.epochs, self.max_epochs, err_train_epoch_mean,
                            err_val_epoch_mean, err_val_best,
                            100 * (err_val_epoch_mean - err_val_prev) /
                            err_val_prev))

                epoch_inst_count -= self._num_inst
                err_val_prev = err_val_epoch_mean
                err_train_cumulative = 0.0
                err_val_cumulative = 0.0
                self.epochs += 1

        if verbose and self.epochs < self.max_epochs:
            if patience_ticks == self.patience:
                early_stop_message = "patience ({}) ran out".format(
                    self.patience)

            elif err_val_epoch_mean <= self.epsilon:
                early_stop_message = (
                    "epoch average validation loss ({:.4f}) smaller "
                    "than 'epsilon' ({:.4f})".format(err_val_epoch_mean,
                                                     self.epsilon))

            else:
                early_stop_message = "unknown reason"

            print("Early stopped due to {}.".format(early_stop_message))

        if recover_best_weight:
            self.weights = best_weights
            if verbose:
                print(
                    "Recovered best weights with validation loss of {:.6f} "
                    "(current validation loss was {:.6f}, relative change of "
                    "{:.2f}%)".format(
                        err_val_best, err_val_epoch_mean,
                        100 * (err_val_best - err_val_epoch_mean) /
                        err_val_epoch_mean))

        if store_errors:
            self.errors = self.errors[:, :self.epochs]

        return None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = 256,
            max_epochs: int = 1000,
            learning_rate: float = 0.0001,
            validation_frac: float = 0.1,
            reg_rate: float = 0.01,
            epsilon: float = 1e-6,
            patience: int = 10,
            patience_margin: float = 0.001,
            recover_best_weight: bool = True,
            add_bias: bool = True,
            store_errors: bool = False,
            verbose: int = 0,
            random_state: t.Optional[int] = None) -> "SGDClassifier":
        """Use the given train data to optimize the classifier parameters.

        Arguments
        ---------
        X : :obj:`np.ndarray`
            Train instances. Each row is an instance, and each column
            is an attribute.

        y : :obj:`np.ndarray`
            Target attribute. Must be an numerical array, and each class
            must be an integer index starting from 0.

        batch_size : :obj:`int`, optional
            Batch size for each parameter update. The instances are
            sampled at random for every batch, without replacement.

        max_epochs : :obj:`int`, optional
            Maximum number of epochs (one epoch means that every training
            example has been seen once, in expectation.)

        learning_rate : :obj:`float`, optional
            Step size in the direction of the negative gradient of the
            loss function for each parameter update.

        validation_frac : :obj:`float`, optional
            Fraction of the train data that must be separated as a
            validation set. Used for early stopping. If 0, all data is
            used as train data, with no early stopping.

        reg_rate : :obj:`float`, optional
            Scale factor for the regularization value. Zero value means
            no regularization. The regularization used is the L2 (Ridge)
            regularization (sum of element-wise squared ``W``.)

        epsilon : :obj:`float`, optional
            Maximum average batch loss for early stopping.

        patience : :obj:`int`, optional
            Maximum number of epochs past the best current loss value
            (i.e., epochs where the average epoch loss does not get
            better than the best value so far) before early stopping.

        patience_margin : :obj:`float`, optional
            Percentage of improvement of the current average epoch loss
            over the best average epoch loss so far in order to consider
            as a real loss improvement.

        recover_best_weight : :obj:`bool`, optional
            If True, recover the weights with minimal validation loss after
            the training process.

        add_bias : :obj:`bool`, optional
            If True, add a constant column full of 1s as the last column
            of ``X`` data, to fit the `bias` term alongside the other
            parameters. This is called `bias trick`, and helps to simplify
            the calculations.

        store_errors : :obj:`bool`, optional
            If True, store the errors for every mini-batch (in the
            ``errors`` instance attribute.) The first row correspond to
            the training erros, while the second row correspond to the
            validation errors.

        verbose : :obj:`int`, optional
            Set the verbosity level of the fit procedure.

        random_state : :obj:`int`, optional
            If given, set the random seed before any pseudo-random
            number generation.

        Returns
        -------
        self
        """
        if not 0 < batch_size <= y.size:
            batch_size = y.size

        _val_data_size = int(y.size * validation_frac)

        if not 0 <= validation_frac < 1:
            raise ValueError("'validation_frac must be in [0, 1) (got "
                             "{}.)".format(validation_frac))

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

        if self._check_data_func is not None:
            self._check_data_func(X, y)

        self._num_classes = np.unique(y).size

        if self._num_classes == 2:
            self._num_classes -= 1

        X_train, X_val, y_train, y_val = X, y, None, None

        if validation_frac > 0.0:
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
                X,
                y,
                shuffle=True,
                stratify=y,
                test_size=validation_frac,
                random_state=random_state)

        self._num_inst, self._num_attr = X_train.shape
        self._data_mean = np.mean(X_train, axis=0)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.reg_rate = reg_rate
        self.patience = patience
        self.epsilon = epsilon
        self.epochs = 0

        if add_bias:
            X_train = self._add_bias(X_train)

            if X_val is not None:
                X_val = self._add_bias(X_val)

            self._data_mean = np.hstack((self._data_mean, 0.0))

        else:
            self._data_mean[-1] = 0.0

        X_train = X_train - self._data_mean
        X_val = X_val - self._data_mean

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(self._num_classes, 1 + self._num_attr))

        self._optimize(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose,
            store_errors=store_errors,
            recover_best_weight=recover_best_weight)

        return self

    def _predict(self,
                 X: np.ndarray,
                 center_data: bool = True,
                 add_bias: bool = True) -> np.ndarray:
        """Calculate the linear scores based on fitted weights."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if add_bias:
            X = self._add_bias(X)

        # Center data based on the training data mean
        if center_data:
            X = X - self._data_mean

        scores = np.dot(self.weights, X.T)

        return scores

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """Calculate the linear scores based on fitted weights."""
        return self._predict(X=X, add_bias=add_bias)


class SoftmaxClassifier(SGDClassifier):
    """Softmax Classifier algorithm.

    It is a generalization of the Logistic Regression classifier
    algorithm for multi-class classification problems.
    """

    def __init__(self):
        """Init a SGD classifier model with cross entropy loss."""
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
        """Linear predictions with Softmax normalization."""
        scores = super()._predict(X=X, add_bias=add_bias)
        scores_norm = self.softmax(scores, axis=0)
        return np.argmax(scores_norm, axis=0)

    def cross_ent_grad(self,
                       X: np.ndarray,
                       y_inds: np.ndarray,
                       scores: t.Optional[np.ndarray] = None,
                       add_bias: bool = True) -> np.ndarray:
        """Cross entropy loss function gradient."""
        if scores is None:
            scores = super()._predict(
                X=X, center_data=False, add_bias=add_bias)

        _scores = self.softmax(scores=scores, axis=0)

        correct_class_ind = np.zeros((self._num_classes, y_inds.size))
        correct_class_ind[y_inds, np.arange(y_inds.size)] = 1

        loss_total = np.dot(_scores - correct_class_ind, X)

        return loss_total / y_inds.size


class SupportVectorClassifier(SGDClassifier):
    """Support Vector Machine with Linear Kernel."""

    def __init__(self):
        """Init a SGD classifier with Hinge loss function."""
        super().__init__(
            func_loss=losses.hinge_loss, func_loss_grad=self.hinge_loss_grad)

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """Predict the class of new instances."""
        return np.argmax(super()._predict(X=X, add_bias=add_bias), axis=0)

    def hinge_loss_grad(self,
                        X: np.ndarray,
                        y_inds: np.ndarray,
                        scores: t.Optional[np.ndarray] = None,
                        add_bias: bool = True,
                        delta: float = 1.0) -> np.ndarray:
        """Gradient of the Hinge loss function."""
        if scores is None:
            scores = super()._predict(
                X=X, center_data=False, add_bias=add_bias)

        _inst_inds = np.arange(y_inds.size)

        correct_class_score = scores[y_inds, _inst_inds]

        # Incorrect classes
        _aux = (scores - correct_class_score + delta > 0).astype(np.float64)

        # Correct classes
        _aux[y_inds, _inst_inds] = 0
        _aux[y_inds, _inst_inds] = -np.sum(_aux, axis=0)

        loss_total = np.dot(_aux, X)

        return loss_total / y_inds.size


class LogisticRegressionClassifier(SGDClassifier):
    def __init__(self):
        """Init a SGD classifier with (negative) Log-Likelihood loss function."""
        super().__init__(
            func_loss=losses.log_likelihood,
            func_loss_grad=self.log_likelihood_grad,
            check_data_func=self._check_binary_problem)

    @staticmethod
    def _check_binary_problem(X: np.ndarray, y: np.ndarray) -> None:
        y_unique = np.unique(y)

        if y_unique.size != 2:
            raise ValueError("'y' must be binary (got {} classes.)".format(
                y_unique.size))

        if not np.all(np.isin(y_unique, [0, 1])):
            raise ValueError("'y' classes must be '0' and '1' "
                             "(got '{}' and '{}'.)".format(*y_unique))

    @classmethod
    def sigmoid(cls, X: np.ndarray) -> np.ndarray:
        """Sigmoid function implementation in a numeric stable manner."""
        sig_vals = np.zeros(X.shape, dtype=np.float64)

        _vals_inst_neg = X < 0
        _vals_inst_pos = ~_vals_inst_neg

        _exp_X = np.exp(X[_vals_inst_neg])

        # Separating in cases for numeric stability
        sig_vals[_vals_inst_neg] = _exp_X / (1.0 + _exp_X)
        sig_vals[_vals_inst_pos] = 1.0 / (1.0 + np.exp(-X[_vals_inst_pos]))

        return sig_vals

    def predict(self,
                X: np.ndarray,
                add_bias: bool = True,
                return_scores: bool = False) -> np.ndarray:
        """Linear predictions with Logistic normalization."""
        scores = super()._predict(X=X, add_bias=add_bias)

        if return_scores:
            return self.sigmoid(scores)

        # Note: no need to apply logistic function, as
        # self.sigmoid(x) > 0.5 iff x > 0.
        return scores.ravel() > 0.0

    def log_likelihood_grad(self,
                            X: np.ndarray,
                            y_inds: np.ndarray,
                            scores: t.Optional[np.ndarray] = None,
                            add_bias: bool = True) -> np.ndarray:
        """Gradient of the (negative) Log-likelihood loss function."""
        if scores is None:
            scores = super()._predict(
                X=X, center_data=False, add_bias=add_bias)

        return -np.dot(y_inds - self.sigmoid(scores), X) / y_inds.size


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


def _plot(X_train: np.ndarray,
          X_test: np.ndarray,
          y_train: np.ndarray,
          y_test: np.ndarray,
          model: SGDClassifier,
          num_bg_inst: int = 75) -> None:
    import matplotlib.pyplot as plt

    null_model_accuracy = np.max(np.bincount(y_train)) / y_train.size
    preds = model.predict(X_test)
    correctly_classified = preds == y_test
    accuracy = np.sum(correctly_classified) / y_test.size
    lims = np.quantile(np.vstack((X_train, X_test)), (0, 1), axis=0)

    A, B = np.meshgrid(
        np.linspace(*lims[:, 0], num_bg_inst),
        np.linspace(*lims[:, 1], num_bg_inst))
    C = model.predict(np.hstack((A.reshape(-1, 1), B.reshape(-1, 1)))).reshape(
        num_bg_inst, num_bg_inst)

    plt.suptitle(
        "Step size: {:.6f} - Accuracy: {:.3f} - Null model accuracy: {:.3f}".
        format(model.learning_rate, accuracy, null_model_accuracy))

    plt.subplot(1, 2, 1)
    plt.title("Instances Scatter plot")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(A, B, c=C, marker="s")
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        edgecolors="black",
        marker=".")

    edge_colors = np.array(
        ["green" if cor_cls else "red" for cor_cls in correctly_classified])

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        edgecolors=edge_colors,
        marker="X")

    plt.subplot(1, 2, 2)
    plt.title("Loss per epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(model.errors[0, :], linestyle="-", color="red", label="Train")
    plt.plot(
        model.errors[1, :], linestyle="-.", color="blue", label="Validation")
    plt.legend()

    plt.show()


def _test_classifier(X: np.ndarray,
                     y: np.ndarray,
                     model: SGDClassifier,
                     learning_rate: t.Optional[float] = None,
                     max_epochs: int = 1000,
                     train_patience: int = 5,
                     batch_size: int = 32,
                     reg_rate: float = 0.01,
                     plot: bool = True) -> None:
    """Full experiment with the implemented SoftmaxClassifier."""
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=32)

    if learning_rate is None:
        folds = sklearn.model_selection.StratifiedKFold(n_splits=5)

        learning_rate_candidates = np.logspace(-2, 0, 15)
        learning_rate_acc = np.zeros(learning_rate_candidates.size)

        for l_ind, learning_rate in enumerate(learning_rate_candidates, 1):
            print("Checking {} out of {} learning rates: {}...".format(
                l_ind, learning_rate_candidates.size, learning_rate))
            for ind, (inds_train, inds_test) in enumerate(
                    folds.split(X_train, y_train)):
                model.fit(
                    X_train[inds_train, :],
                    y_train[inds_train],
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=train_patience,
                    reg_rate=reg_rate,
                    learning_rate=learning_rate)

                preds = model.predict(X_train[inds_test, :])
                learning_rate_acc[ind] += np.sum(
                    preds == y_train[inds_test]) / inds_test.size

        learning_rate = learning_rate_candidates[np.argmax(learning_rate_acc)]
        print("Best learning rate:", learning_rate)

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        verbose=1,
        patience=train_patience,
        max_epochs=max_epochs,
        store_errors=plot,
        reg_rate=reg_rate,
        epsilon=0,
        learning_rate=learning_rate)

    print("Accuracy:", np.sum(model.predict(X_test) == y_test) / y_test.size)

    if plot:
        _plot(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model=model)


def _test_softmax_classifier_01() -> None:
    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    _test_classifier(model=SoftmaxClassifier(), X=X, y=y, learning_rate=0.014)


def _test_softmax_classifier_02() -> None:
    from sklearn.datasets import load_iris
    iris = load_iris()
    _test_classifier(
        model=SoftmaxClassifier(),
        learning_rate=0.25,
        batch_size=50,
        X=iris.data[:, [0, 1]],
        y=iris.target,
        train_patience=5)


def _test_softmax_classifier_03() -> None:
    from sklearn.datasets import load_iris
    iris = load_iris()
    _test_classifier(
        model=SoftmaxClassifier(),
        X=iris.data,
        y=iris.target,
        learning_rate=0.25,
        batch_size=50,
        train_patience=20,
        reg_rate=0.1,
        plot=False)


def _test_softmax_grad() -> None:
    import gradient_check

    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    X = np.hstack((X, np.ones((y.size, 1))))

    func = lambda W: losses.cross_ent_loss(
        X=X,
        y_inds=y,
        W=W.reshape((3, 2 + 1)))

    model = SoftmaxClassifier()
    model.fit(X, y, max_epochs=0, add_bias=False)

    def func_grad(W: np.ndarray):
        model.weights = W.reshape((3, 2 + 1))
        return model.cross_ent_grad(X=X, y_inds=y, add_bias=False).ravel()

    error = gradient_check.gradient_check(
        func=func,
        analytic_grad=func_grad,
        x_limits=np.array([-5, 5] * 9).reshape(-1, 2),
        num_it=2000,
        random_state=32,
        verbose=1)


def _test_hinge_grad() -> None:
    import gradient_check

    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    X = np.hstack((X, np.ones((y.size, 1))))

    func = lambda W: losses.hinge_loss(X=X, y_inds=y, W=W.reshape((3, 2 + 1)))

    model = SupportVectorClassifier()
    model.fit(X, y, max_epochs=0, add_bias=False)

    def func_grad(W: np.ndarray):
        model.weights = W.reshape((3, 2 + 1))
        return model.hinge_loss_grad(X=X, y_inds=y, add_bias=False).ravel()

    error = gradient_check.gradient_check(
        func=func,
        analytic_grad=func_grad,
        x_limits=np.array([-5, 5] * 9).reshape(-1, 2),
        num_it=2000,
        random_state=32,
        verbose=1)


def _test_log_likelihood_grad() -> None:
    import gradient_check

    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    # Make a binary classification problem
    X = X[:-inst_per_class, :]
    y = y[:-inst_per_class]

    X = np.hstack((X, np.ones((y.size, 1))))

    func = lambda W: losses.log_likelihood(X=X, y_inds=y, W=W.reshape((1, 2 + 1)))

    model = LogisticRegressionClassifier()
    model.fit(X, y, max_epochs=0, add_bias=False)

    def func_grad(W: np.ndarray):
        model.weights = W.reshape((1, 2 + 1))
        return model.log_likelihood_grad(X=X, y_inds=y, add_bias=False).ravel()

    error = gradient_check.gradient_check(
        func=func,
        analytic_grad=func_grad,
        x_limits=np.array([-1, 1] * 3).reshape(-1, 2),
        num_it=3000,
        random_state=32,
        verbose=1)


def _test_svc_01() -> None:
    np.random.seed(16)

    inst_per_class = 512
    X, y = _gen_data(inst_per_class=inst_per_class)

    _test_classifier(
        model=SupportVectorClassifier(),
        learning_rate=0.01,
        batch_size=128,
        train_patience=5,
        X=X,
        y=y)


def _test_svc_02() -> None:
    from sklearn.datasets import load_iris
    iris = load_iris()
    _test_classifier(
        model=SupportVectorClassifier(),
        X=iris.data[:, [0, 1]],
        y=iris.target,
        batch_size=75,
        learning_rate=0.1,
        train_patience=10)


def _test_svc_03() -> None:
    from sklearn.datasets import load_iris
    iris = load_iris()
    _test_classifier(
        model=SupportVectorClassifier(),
        X=iris.data,
        y=iris.target,
        reg_rate=0.05,
        learning_rate=0.02,
        train_patience=20,
        plot=False)


def _test_logistic_classifier_01() -> None:
    np.random.seed(16)

    inst_per_class = 200
    X, y = _gen_data(inst_per_class=inst_per_class)

    X = X[:-inst_per_class, :]
    y = y[:-inst_per_class]

    _test_classifier(
        model=LogisticRegressionClassifier(),
        X=X,
        y=y,
        train_patience=5,
        learning_rate=0.01)


if __name__ == "__main__":
    # _test_softmax_grad()
    # _test_hinge_grad()
    # _test_log_likelihood_grad()

    _test_softmax_classifier_01()
    _test_svc_01()
    _test_logistic_classifier_01()

    _test_softmax_classifier_02()
    _test_svc_02()

    _test_softmax_classifier_03()
    _test_svc_03()
