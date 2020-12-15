import typing as t
import warnings

import numpy as np


def issue_unused_args_warnings(unused_args: t.Dict[str, t.Any]) -> None:
    for arg in unused_args:
        warnings.warn(f"Unused learning rate decay argument: {arg}.", UserWarning)


def solve_lr_update(lr_update: str) -> t.Tuple[t.Callable, t.Callable]:
    return LR_UPDATE[lr_update]


def init_constant(learning_rate: float, **kwargs) -> t.Dict[str, t.Any]:
    assert learning_rate > 0.0

    cache = dict()
    cache["lr"] = learning_rate

    issue_unused_args_warnings(kwargs)

    return cache


def update_constant(cache: t.Dict[str, t.Any]) -> float:
    return cache["lr"]


def init_inv(
    learning_rate: float, decay_rate: float = 1.0, **kwargs
) -> t.Dict[str, t.Any]:
    assert learning_rate > 0.0
    assert decay_rate > 0.0

    cache = dict()
    cache["lr_initial"] = learning_rate
    cache["decay_rate"] = decay_rate
    cache["epoch_num"] = 0

    issue_unused_args_warnings(kwargs)

    return cache


def update_inv(cache: t.Dict[str, t.Any]) -> float:
    lr_initial = cache["lr_initial"]
    decay_rate = cache["decay_rate"]
    epoch_num = cache["epoch_num"]

    cache["epoch_num"] += 1

    return lr_initial / (1.0 + decay_rate * (1 + epoch_num))


def init_exp(
    learning_rate: float, decay_rate: float = 0.99, **kwargs
) -> t.Dict[str, t.Any]:
    assert learning_rate > 0.0
    assert 0.0 < decay_rate < 1.0

    cache = dict()
    cache["lr_initial"] = learning_rate
    cache["decay_rate"] = decay_rate
    cache["epoch_num"] = 0

    issue_unused_args_warnings(kwargs)

    return cache


def update_exp(cache: t.Dict[str, t.Any]) -> float:
    lr_initial = cache["lr_initial"]
    decay_rate = cache["decay_rate"]
    epoch_num = cache["epoch_num"]

    cache["epoch_num"] += 1

    return lr_initial * decay_rate ** (1 + epoch_num)


def init_inv_sqrt(
    learning_rate: float, constant: float = 1.0, **kwargs
) -> t.Dict[str, t.Any]:
    assert learning_rate > 0.0
    assert constant > 0.0

    cache = dict()
    cache["lr_initial"] = learning_rate
    cache["constant"] = constant
    cache["epoch_num"] = 0

    issue_unused_args_warnings(kwargs)

    return cache


def update_inv_sqrt(cache: t.Dict[str, t.Any]) -> float:
    lr_initial = cache["lr_initial"]
    constant = cache["constant"]
    epoch_num = cache["epoch_num"]

    cache["epoch_num"] += 1

    return lr_initial * constant / np.sqrt(1 + epoch_num)


def init_disc_frac(
    learning_rate: float, decay_rate: float = 0.5, epochs: int = 32, **kwargs
) -> t.Dict[str, t.Any]:
    assert learning_rate > 0.0
    assert epochs > 0.0
    assert 0.0 < decay_rate < 1.0

    cache = dict()
    cache["lr"] = learning_rate
    cache["epochs"] = epochs
    cache["decay_rate"] = decay_rate
    cache["epoch_num"] = 0

    issue_unused_args_warnings(kwargs)

    return cache


def update_disc_frac(cache: t.Dict[str, t.Any]) -> float:
    lr = cache["lr"]
    epochs = cache["epochs"]
    epoch_num = cache["epoch_num"]
    decay_rate = cache["decay_rate"]

    cache["epoch_num"] += 1

    if (1 + epoch_num) % epochs == 0:
        cache["epoch_num"] = epoch_num = 0
        cache["lr"] = lr = lr * decay_rate

    return lr


LR_UPDATE = {
    "constant": (init_constant, update_constant),
    "exp": (init_exp, update_exp),
    "inv": (init_inv, update_inv),
    "inv_sqrt": (init_inv_sqrt, update_inv_sqrt),
    "disc_frac": (init_disc_frac, update_disc_frac),
}  # type: t.Dict[str, t.Tuple[t.Callable, t.Callable]]
