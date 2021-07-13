"""Perform linear regression using SVD."""
import numpy as np


def lin_reg_with_svd():
    import sklearn.datasets
    import sklearn.linear_model

    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    X = np.column_stack((np.ones_like(y), X))

    X_pinv = np.linalg.pinv(X)
    coeffs_a = np.dot(X_pinv, y)

    lin_model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    lin_model.fit(X, y)
    coeffs_b = lin_model.coef_

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # X = U S Vt -> X^{+} = V S^{-1} Ut
    # U : (n, m)
    # S : (m, m)
    # Vt: (m, m)
    # n >> m

    # (US) Vt = (n * m * m) + (n * m * m)
    # U(S Vt) = (m * m * m) + (n * m * m)
    X_pinv_manual = np.dot(np.dot(Vt.T, np.diag(1 / S)), U.T)
    coeffs_c = np.dot(X_pinv_manual, y)

    assert np.allclose(coeffs_a, coeffs_b)
    assert np.allclose(coeffs_b, coeffs_c)


if __name__ == "__main__":
    lin_reg_with_svd()
