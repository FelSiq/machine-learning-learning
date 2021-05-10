import numpy as np


class LogReg:
    def __init__(
        self,
        max_epochs: int = 512,
        learning_rate: float = 1e-1,
        batch_size: int = 32,
        add_intercept: bool = True,
        epsilon: float = 1e-3,
    ):
        assert int(max_epochs) >= 1
        assert float(learning_rate) > 0.0
        assert int(batch_size) >= 1

        self.coeffs = np.empty(0)
        self.add_intercept = add_intercept
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)

        self._training = False

    @staticmethod
    def _sigmoid(x):
        pos_inds = x >= 0
        neg_inds = ~pos_inds

        neg_exp = np.exp(x[neg_inds])

        res = np.zeros_like(x)

        res[pos_inds] = 1.0 / (1.0 + np.exp(-x[pos_inds]))
        res[neg_inds] = neg_exp / (1.0 + neg_exp)

        return res

    def predict(self, X):
        n, _ = X.shape

        if self.add_intercept and not self._training:
            X = np.column_stack((np.ones(n, dtype=float), X))

        y_preds = self._sigmoid(np.dot(X, self.coeffs))

        return y_preds

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asfarray(y).ravel()

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, m = X.shape

        X = np.column_stack((np.ones(n, dtype=float), X))

        self.coeffs = np.random.randn(m + self.add_intercept)

        self._training = True

        for i in np.arange(1, 1 + self.max_epochs):
            max_diff = -np.inf

            for j in np.arange(0, y.size, self.batch_size):
                X_batch = X[j : j + self.batch_size, :]
                y_batch = y[j : j + self.batch_size]

                y_preds = self.predict(X_batch)

                grad = np.dot(X_batch.T, y_preds - y_batch)
                update = self.learning_rate * grad / self.batch_size
                self.coeffs -= update

                if self.epsilon <= 0:
                    max_diff = np.inf
                    continue

                inf_norm_diff = np.linalg.norm(update, ord=np.inf)
                max_diff = max(max_diff, inf_norm_diff)

            if max_diff < self.epsilon:
                print(f"Break loop in {i} epoch.")
                break

        self._training = False

        return self


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.linear_model
    import sklearn.preprocessing

    model = LogReg(learning_rate=1e-1, epsilon=1e-5, batch_size=128)

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    n_splits = 10

    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    ref = sklearn.linear_model.LogisticRegression(
        max_iter=512, penalty="none", verbose=0
    )

    acc_train = acc_eval = 0.0
    acc_train_ref = acc_eval_ref = 0.0

    scaler = sklearn.preprocessing.StandardScaler()

    def acc(a, b):
        return sklearn.metrics.accuracy_score(a, b)

    for inds_train, inds_eval in splitter.split(X, y):
        X_train, X_eval = X[inds_train, :], X[inds_eval, :]
        y_train, y_eval = y[inds_train], y[inds_eval]

        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        model.fit(X_train, y_train)
        ref.fit(X_train, y_train)

        y_preds = model.predict(X_eval) > 0.5
        y_preds_ref = ref.predict(X_eval)
        acc_eval += acc(y_preds, y_eval)
        acc_eval_ref += acc(y_preds_ref, y_eval)

        y_preds = model.predict(X_train) > 0.5
        y_preds_ref = ref.predict(X_train)
        acc_train += acc(y_preds, y_train)
        acc_train_ref += acc(y_preds_ref, y_train)

    acc_eval /= n_splits
    acc_train /= n_splits
    acc_eval_ref /= n_splits
    acc_train_ref /= n_splits

    _, cls_freqs = np.unique(y, return_counts=True)
    baseline = np.max(cls_freqs) / y.size
    print(f"Baseline              : {baseline:.3f}")
    print(f"(mine) Train accuracy : {acc_train:.3f}")
    print(f"(mine) Eval accuracy  : {acc_eval:.3f}")
    print(f"(ref)  Train accuracy : {acc_train_ref:.3f}")
    print(f"(ref)  Eval accuracy  : {acc_eval_ref:.3f}")


if __name__ == "__main__":
    _test()
