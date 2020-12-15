"""Implement different types of regularizations."""
import numpy as np


def l2(W: np.ndarray, lambda_: float = 0.01,
       exclude_bias: bool = False) -> float:
    """Ridge (L2) regularization.

    It is defined as the sum of element-wise squared weights.

    It has the property of encouraging models with distributed
    power between a large amount of evenly-distributed parameters.

    Arguments
    ---------
    W : :obj:`np.ndarray`
        Array of weights.

    lambda_ : :obj:`float`, optional
        Regularization power. If 0, no regularization is applied.
        The larger this value is, the more regularization is
        applied.

    exclude_bias : :obj:`bool`, optional
        If True, exclude the last column in the regularization
        calculation (it is assumed to be the bias column - a column
        full of 1s), concept known as `bias trick` to simplify
        calculations.

    Returns
    -------
    float
        Ridge (L2) regularization value.
    """
    reg_factor = 0

    if not np.equal(0, lambda_):
        if exclude_bias:
            W = W[:, :-1]

        reg_factor = lambda_ * np.sum(np.square(W))

    return reg_factor


def l2_grad(W: np.ndarray, lambda_: float = 0.01,
            exclude_bias: bool = False) -> np.ndarray:
    """Gradient of the Ridge (L2) regularization."""
    if exclude_bias:
        W = np.hstack((W[:, :-1], np.zeros((W.shape[0], 1))))

    return 2.0 * lambda_ * W
