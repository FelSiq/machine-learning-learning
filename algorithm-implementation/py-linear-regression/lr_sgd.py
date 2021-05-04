import numpy as np


class LinReg:
    """Implements Linear Regression with SGD."""

    def __init__(
        self,
        max_epochs: int = 512,
        learning_rate: float = 1e-1,
        batch_size: int = 32,
        add_intercept: bool = True,
        epsilon: float = 1e-3,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
    ):
        assert int(max_epochs) >= 1
        assert float(learning_rate) > 0.0
        assert int(batch_size) >= 1
        assert float(l1_reg) >= 0.0
        assert float(l2_reg) >= 0.0

        self.coeffs = np.empty(0)
        self.add_intercept = add_intercept
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)

        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)

        self._training = False

    def predict(self, X):
        n, _ = X.shape

        if self.add_intercept and not self._training:
            X = np.column_stack((np.ones(n, dtype=float), X))

        y_preds = np.dot(X, self.coeffs)

        return y_preds

    def _calc_reg(self, m):
        reg = np.zeros(m + self.add_intercept, dtype=float)
        reg += self.l2_reg * self.coeffs
        reg += self.l1_reg * np.sign(self.coeffs)
        reg[: self.add_intercept] = 0.0
        return reg

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

                reg = self._calc_reg(m)
                grad = np.dot(X_batch.T, y_preds - y_batch) + reg
                update = self.learning_rate * grad / self.batch_size
                self.coeffs -= update

                if self.epsilon <= 0:
                    continue

                inf_norm_diff = np.linalg.norm(update, ord=np.inf)
                max_diff = max(max_diff, inf_norm_diff)

            if max_diff < self.epsilon:
                print(f"Break loop in {i} epoch.")
                break

        self._training = False

        return self


def _test():
    import test

    test.test(model=LinReg())


if __name__ == "__main__":
    _test()
