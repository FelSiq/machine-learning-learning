"""Learning rate decay functions."""
import typing as t

import numpy as np

_LRType = t.Union[float, np.ndarray]


def identity(
    learning_rate: _LRType, decay_rate: float = 0.0, epoch_num: int = 0
) -> _LRType:
    """Do not change the learning rate.

    Arguments
    ---------
    learning_rate : :obj:`float` or :obj:`np.ndarray`
        Learning rate or array with a group of learning rates.

    decay_rate : :obj:`float`
        Used only to keep consistency of the learning rate
        decay API.

    epoch_num : :obj:`int`
        Used only to keep consistency of the learning rate
        decay API.

    Return
    ------
    float
        Learning rate.
    """
    return learning_rate


def step(
    learning_rate: _LRType, decay_rate: float, epoch_num: int, step_delay: int = 10
) -> _LRType:
    """Reduce learning rate after every ``step_delay`` epochs.

    Arguments
    ---------
    learning_rate : :obj:`float` or :obj:`np.ndarray`
        Learning rate or array with a group of learning rates.

    decay_rate : :obj:`float`
        Decay rate for each learning rate.

    epoch_num : :obj:`int`
        Number of epochs so far.

    step_delay : :obj:`int`, optional
        Interval in epochs between each learning rate decrease.

    Return
    ------
    float
        New learning rate.
    """
    if (epoch_num + 1) % step_delay == 0:
        return decay_rate * learning_rate

    return learning_rate


def val_step(
    learning_rate: _LRType,
    decay_rate: float,
    loss_val_cur: float,
    loss_val_prev: float,
    min_diff_frac: float = 0.01,
) -> _LRType:
    r"""Decay the learning rate based on validation loss.

    If \frac{loss_val_cur}{loss_val_prev} < (1 - min_diff_frac),
    then the learning_rate is decreased by a factor of
    ``decay_rate``.

    Arguments
    ---------
    learning_rate : :obj:`float` or :obj:`np.ndarray`
        Learning rate or array with a group of learning rates.

    decay_rate : :obj:`float`
        Decay rate for each learning rate.

    loss_val_cur : :obj:`float`
        Current validation set loss.

    loss_val_prev : :obj:`float`
        Previous validation set loss.

    min_diff_frac : :obj:`float`, optional
        ``loss_val_cur`` C is considered different from ``loss_val_prev``
        P if and only if: C < (1 - min_diff_frac) * P. Otherwise, the
        learning rate is not decreased.

    Return
    ------
    float
        New learning rate.
    """
    if loss_val_cur < (1 - min_diff_frac) * loss_val_prev:
        return decay_rate * learning_rate

    return learning_rate


def exp(learning_rate: _LRType, decay_rate: float, epoch_num: int) -> _LRType:
    """Exponential learning rate decay.

    Arguments
    ---------
    learning_rate : :obj:`float` or :obj:`np.ndarray`
        Learning rate or array with a group of learning rates.

    decay_rate : :obj:`float`
        Decay rate for each learning rate.

    epoch_num : :obj:`int`
        Number of epochs so far.

    Return
    ------
    float
        New learning rate.
    """
    return learning_rate * np.exp(-decay_rate * epoch_num)


def inv(learning_rate: _LRType, decay_rate: float, epoch_num: int) -> _LRType:
    """Learning rate decay inversely proportional to ``epoch_num``.

    Arguments
    ---------
    learning_rate : :obj:`float` or :obj:`np.ndarray`
        Learning rate or array with a group of learning rates.

    decay_rate : :obj:`float`
        Decay rate for each learning rate.

    epoch_num : :obj:`int`
        Number of epochs so far.

    Return
    ------
    float
        New learning rate.
    """
    return learning_rate / (1 + decay_rate * epoch_num)
