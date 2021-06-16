import numpy as np


class LinReg:
    """Implements Linear Regression with SVD."""

    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
        self.coeffs = np.empty(0)

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asfarray(y)

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pseudo_inv = Vt.T @ np.diag(1.0 / S) @ U.T
        self.coeffs = np.dot(pseudo_inv, y)

        return self

    def predict(self, X):
        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        preds = np.dot(X, self.coeffs)

        return preds


def _test():
    import test

    test.test(model=LinReg())


if __name__ == "__main__":
    _test()
