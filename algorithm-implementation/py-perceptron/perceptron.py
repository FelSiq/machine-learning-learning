import numpy as np

import activations


class PerceptronClassifier:
    def __init__(
        self,
        learning_rate: float = 1e-2,
        max_epochs: int = 128,
        batch_size: int = 32,
        activation: str = "sigmoid",
        add_intercept: bool = True,
        epsilon: float = 1e-2,
    ):
        assert int(max_epochs) >= 1
        assert int(batch_size) >= 1
        assert float(learning_rate) > 0.0
        assert float(epsilon) >= 0.0
        assert activation in {"sigmoid", "relu", "tanh"}

        self.weights = np.empty(0)
        self.add_intercept = add_intercept
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)
        self.activation = activation
        self._activation_fun, self._activation_grad = activations.get_activation(
            activation
        )

        self._training = False

    def _run_epoch(self, X, y, batch_inds):
        max_diff = -np.inf

        for start in np.arange(0, y.size, self.batch_size):
            end = start + self.batch_size

            X_batch = X[batch_inds[start:end], :]
            y_batch = y[batch_inds[start:end]]

            y_preds = self.predict(X_batch)

            grad = X_batch.T @ (
                self._activation_grad(y_preds)
                * (y_preds - y_batch)
                / (1e-6 + y_preds * (1.0 - y_preds))
            )

            update = self.learning_rate * grad / self.batch_size
            self.weights -= update

            if self.epsilon <= 0.0:
                max_diff = np.inf
                continue

            max_diff = max(max_diff, np.linalg.norm(update, ord=np.inf))

        return max_diff

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asarray(y)

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, m = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        self.weights = np.random.randn(m + 1)
        self._training = True
        batch_inds = np.arange(n)

        for i in np.arange(1, 1 + self.max_epochs):
            np.random.shuffle(batch_inds)
            max_diff = self._run_epoch(X, y, batch_inds)

            if max_diff < self.epsilon:
                print(f"Early stopped at epoch {i}.")
                break

        self._training = False

        return self

    def predict(self, X):
        X = np.asfarray(X)
        n, _ = X.shape

        if not self._training and self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        preds = self._activation_fun(X @ self.weights)

        return preds


def _test():
    import sklearn.model_selection
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.preprocessing
    import sklearn.linear_model
    import matplotlib.pyplot as plt

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    print(X.shape)

    model = PerceptronClassifier(
        batch_size=64,
        learning_rate=5e-2,
        epsilon=1e-4,
        activation="relu",
        max_epochs=32,
    )

    ref = sklearn.linear_model.Perceptron(max_iter=1024)
    n_splits = 10
    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )
    scaler = sklearn.preprocessing.StandardScaler()

    train_err = eval_err = 0.0
    train_err_ref = eval_err_ref = 0.0

    for inds_train, inds_eval in splitter.split(X, y):
        X_train, X_eval = X[inds_train, :], X[inds_eval, :]
        y_train, y_eval = y[inds_train], y[inds_eval]

        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        model.fit(X_train, y_train)
        ref.fit(X_train, y_train)

        y_preds_train = model.predict(X_train)
        y_preds_eval = model.predict(X_eval)

        threshold = np.mean(y_train)
        y_preds_train = y_preds_train > threshold
        y_preds_eval = y_preds_eval > threshold

        y_preds_train = y_preds_train.astype(int, copy=False)
        y_preds_eval = y_preds_eval.astype(int, copy=False)

        y_preds_train_ref = ref.predict(X_train)
        y_preds_eval_ref = ref.predict(X_eval)

        train_err += sklearn.metrics.accuracy_score(y_preds_train, y_train)
        eval_err += sklearn.metrics.accuracy_score(y_preds_eval, y_eval)
        train_err_ref += sklearn.metrics.accuracy_score(y_preds_train_ref, y_train)
        eval_err_ref += sklearn.metrics.accuracy_score(y_preds_eval_ref, y_eval)

    train_err /= n_splits
    eval_err /= n_splits
    train_err_ref /= n_splits
    eval_err_ref /= n_splits

    print(f"(mine) Train err : {train_err:.3f}")
    print(f"(mine) Eval err  : {eval_err:.3f}")
    print(f"(ref)  Train err : {train_err_ref:.3f}")
    print(f"(ref)  Eval err  : {eval_err_ref:.3f}")


if __name__ == "__main__":
    _test()
