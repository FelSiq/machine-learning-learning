"""Implement different types of regularizations."""
import numpy as np


def l2(W: np.ndarray, lambda_: float = 0.0001) -> float:
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

    Returns
    -------
    float
        Ridge (L2) regularization value.
    """
    reg_factor = 0

    if not np.equal(0, lambda_):
        reg_factor = lambda_ * np.sum(np.square(W))

    return reg_factor


def l2_grad(W: np.ndarray, lambda_: float = 0.0001) -> np.ndarray:
    """Gradient of the Ridge (L2) regularization."""
    return 2.0 * lambda_ * W
