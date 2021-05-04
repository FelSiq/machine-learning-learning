import numpy as np


class LinReg:
    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asfarray(y)

        assert X.ndim == 2
        assert X.shape[0] == y.size

        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        self.coeffs = np.linalg.inv(X.T @ X) @ X.T @ y

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
