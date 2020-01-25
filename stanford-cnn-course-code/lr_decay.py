"""Learning rate decay functions."""
import typing as t

import numpy as np

_LRType = t.Union[float, np.ndarray]


def identity(learning_rate: _LRType, decay_rate: float,
             epoch_num: int) -> _LRType:
    """Do not change the learning rate."""
    return learning_rate


def step(learning_rate: _LRType,
         decay_rate: float,
         epoch_num: int,
         step_delay: int = 10) -> _LRType:
    """Reduce learning rate after every ``step_delay`` epochs."""
    if (epoch_num + 1) % step_delay == 0:
        return decay_rate * learning_rate

    return learning_rate


def exp(learning_rate: _LRType, decay_rate: float, epoch_num: int) -> _LRType:
    """Exponential learning rate decay."""
    return learning_rate * np.exp(-decay_rate * epoch_num)


def inv(learning_rate: _LRType, decay_rate: float, epoch_num: int) -> _LRType:
    """Learning rate decay inversely proportional to ``epoch_num``."""
    return learning_rate / (1 + decay_rate * epoch_num)
