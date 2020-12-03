import typing as t
import warnings

import numpy as np


def issue_unused_args_warnings(unused_args: t.Dict[str, t.Any]) -> None:
    for arg in unused_args:
        warnings.warn(f"Unused optimizer argument: {arg}.", UserWarning)


def solve_optim(optimizer: str) -> t.Tuple[t.Callable, t.Callable]:
    return OPTIM[optimizer]


def init_gd(params: t.Dict[str, np.ndarray], **kwargs) -> t.Dict[str, np.ndarray]:
    cache = dict()
    issue_unused_args_warnings(kwargs)
    return cache


def update_gd(
    grads: t.Dict[str, np.ndarray], cache: t.Dict[str, t.Any]
) -> t.Dict[str, np.ndarray]:
    updates = dict()

    for dparam, cur_grads in grads.items():
        updates["u" + dparam[1:]] = cur_grads

    return updates


def init_momentum(
    params: t.Dict[str, np.ndarray],
    bias_correction: bool = True,
    beta1: float = 0.9,
    **kwargs,
) -> t.Dict[str, np.ndarray]:
    cache = dict()

    assert 0.0 <= beta1 <= 1.0

    for param, cur_params in params.items():
        cache["v" + param] = np.zeros_like(cur_params)

    cache["beta1"] = beta1
    cache["bias_correction"] = bias_correction
    cache["it_num"] = 0

    issue_unused_args_warnings(kwargs)

    return cache


def update_momentum(
    grads: t.Dict[str, np.ndarray], cache: t.Dict[str, t.Any]
) -> t.Dict[str, np.ndarray]:
    bias_correction = cache["bias_correction"]
    beta1 = cache["beta1"]
    it_num = cache["it_num"]

    updates = dict()

    for dparam, cur_grad in grads.items():
        name_v_param = "v" + dparam[1:]
        name_u_param = "u" + dparam[1:]

        if name_v_param not in cache:
            continue

        cur_vel = cache[name_v_param]
        cur_vel = beta1 * cur_vel + (1.0 - beta1) * cur_grad

        if bias_correction and beta1 ** it_num > 1e-8:
            cur_vel /= 1.0 + beta1 ** it_num

        cache[name_v_param] = cur_vel
        cache["it_num"] += 1
        updates[name_u_param] = cur_vel

    return updates


def init_rmsprop(
    params: t.Dict[str, np.ndarray],
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    **kwargs,
) -> t.Dict[str, np.ndarray]:
    cache = dict()

    assert epsilon >= 0.0
    assert 0.0 <= beta2 <= 1.0

    for param, cur_params in params.items():
        cache["s" + param] = np.zeros_like(cur_params)

    cache["beta2"] = beta2
    cache["epsilon"] = epsilon

    issue_unused_args_warnings(kwargs)

    return cache


def update_rmsprop(
    grads: t.Dict[str, np.ndarray], cache: t.Dict[str, t.Any]
) -> t.Dict[str, np.ndarray]:

    beta2 = cache["beta2"]
    epsilon = cache["epsilon"]

    updates = dict()

    for dparam, cur_grad in grads.items():
        name_s_param = "s" + dparam[1:]
        name_u_param = "u" + dparam[1:]

        if name_s_param not in cache:
            continue

        cur_vel_sqr = cache[name_s_param]
        cur_vel_sqr = beta2 * cur_vel_sqr + (1.0 - beta2) * np.square(cur_grad)

        cache[name_s_param] = cur_vel_sqr
        updates[name_u_param] = cur_grad / (np.sqrt(cur_vel_sqr) + epsilon)

    return updates


OPTIM = {
    "gd": (init_gd, update_gd),
    "momentum": (init_momentum, update_momentum),
    "rmsprop": (init_rmsprop, update_rmsprop),
}  # type: t.Dict[str, t.Tuple[t.Callable, t.Callable]]
