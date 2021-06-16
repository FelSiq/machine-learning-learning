import numpy as np


class LinReg:
    def __init__(self, add_intercept: bool = True, l2_reg: float = 0.0):
        assert float(l2_reg) >= 0.0

        self.add_intercept = add_intercept
        self.coeffs = np.empty(0)
        self.l2_reg = float(l2_reg)

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asfarray(y)

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, m = X.shape

        reg = self.l2_reg * np.eye(m + self.add_intercept)

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))
            reg[0, 0] = 0.0

        self.coeffs = np.linalg.inv(X.T @ X + reg) @ X.T @ y

        return self

    def predict(self, X):
        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        preds = np.dot(X, self.coeffs)

        return preds


def _test():
    import test

    test.test(model=LinReg(l2_reg=0.1))


if __name__ == "__main__":
    _test()
