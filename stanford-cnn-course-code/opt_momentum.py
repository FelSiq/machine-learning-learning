"""Momentum and optimization techniques."""
import numpy as np


def momentum_vanilla(momentum: np.ndarray,
                     grad: np.ndarray,
                     learning_rate: float,
                     momentum_rate: float = 0.9) -> np.ndarray:
    """Vanilla momentum.

    Arguments
    ---------
    momentum : :obj:`np.ndarray`, (num_classes, num_attr)
        Previous momentum for each parameter. If there is not
        previous momentum calculated, init as an array full of
        zeros.

    grad : :obj:`np.ndarray`, (num_classes, num_attr)
        Gradient calculated for each of the parameters.

    learning_rate : :obj:`float`
        Multiplicative constant for the gradient.

    momentum_rate : :obj:`float`, optional
        Multiplicative constant for momentum decay.

    Return
    ------
    :obj:`np.ndarray`, (num_classes, num_attr)
        Total change for each of the parameters with vanilla
        momentum.
    """
    # Update momentum in-place
    momentum *= momentum_rate
    momentum += grad

    total_change = learning_rate * (momentum_rate * momentum + grad)

    return total_change


def momentum_nesterov(momentum: np.ndarray,
                      grad: np.ndarray,
                      learning_rate: float,
                      momentum_rate: float = 0.9) -> np.ndarray:
    """Nesterov momentum.

    Arguments
    ---------
    momentum : :obj:`np.ndarray`, (num_classes, num_attr)
        Previous momentum for each parameter. If there is not
        previous momentum calculated, init as an array full of
        zeros.

    grad : :obj:`np.ndarray`, (num_classes, num_attr)
        Gradient calculated for each of the parameters.

    learning_rate : :obj:`float`
        Multiplicative constant for the gradient.

    momentum_rate : :obj:`float`, optional
        Multiplicative constant for momentum decay.

    Return
    ------
    :obj:`np.ndarray`, (num_classes, num_attr)
        Total change for each of the parameters with nesterov
        momentum.
    """
    old_momentum = np.copy(momentum)

    # Update momentum in-place
    momentum *= momentum_rate
    momentum -= learning_rate * grad

    total_change = momentum_rate * old_momentum - (
        1 + momentum_rate) * momentum

    return total_change


def opt_adagrad(grad_sqr: np.ndarray, grad: np.ndarray,
                learning_rate: float) -> np.ndarray:
    """."""
    grad_sqr += np.square(grad)

    total_change = learning_rate * grad / (np.sqrt(grad_sqr) + 1e-7)

    return total_change


def opt_rmsprop(grad_sqr: np.ndarray,
                grad: np.ndarray,
                learning_rate: float,
                decay_rate: float = 0.1) -> np.ndarray:
    """."""
    grad_sqrt *= decay_rate
    grad_sqr += (1 - decay_rate) * np.square(grad)

    total_change = learning_rate * grad / (np.sqrt(grad_sqr) + 1e-7)

    return total_change
