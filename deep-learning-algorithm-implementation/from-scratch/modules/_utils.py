import typing as t

import numpy as np


def all_positive(vals):
    a = not isinstance(vals, int) or vals > 0
    b = not hasattr(vals, "__len__") or all(map(lambda x: x > 0, vals))
    return a and b


def all_nonzero(vals):
    a = not isinstance(vals, int) or vals != 0
    b = not hasattr(vals, "__len__") or all(map(lambda x: x != 0, vals))
    return a and b


def replicate(vals, n):
    if hasattr(vals, "__len__"):
        assert len(vals) == n
        return tuple(vals)

    return tuple([vals] * n)


def collapse(items, cls):
    if isinstance(items, cls):
        return [items]

    cur_items = []

    for item in items:
        cur_items.extend(collapse(item, cls))

    return cur_items


# NOTE: either 'normal' and 'uniform' distributions are initialized
# to have the very same variance.
_WEIGHT_INIT_PARAM = {
    ("normal", "he"): lambda dim_in, _: np.sqrt(2.0 / dim_in),
    ("normal", "xavier"): lambda dim_in, _: np.sqrt(1.0 / (3.0 * dim_in)),
    ("normal", "xavier_norm"): lambda dim_in, dim_out: np.sqrt(
        2.0 / (dim_in + dim_out)
    ),
    ("uniform", "he"): lambda dim_in, _: np.sqrt(6.0 / dim_in),
    ("uniform", "xavier"): lambda dim_in, _: np.sqrt(1.0 / dim_in),
    ("uniform", "xavier_norm"): lambda dim_in, dim_out: np.sqrt(
        6.0 / (dim_in + dim_out)
    ),
}

# Known rules of thumb:
# 'he': suitable for ReLU activations
# 'xavier': suitable for Tanh and Sigmoid activations


def get_weight_init_dist_params(
    std: t.Union[str, float], dist: str, shape: t.Union[int, int]
):
    if np.isreal(std):
        return float(std)

    assert dist in {"normal", "uniform"}
    assert not isinstance(std, str) or std in {"he", "xavier", "xavier_norm"}

    dim_in, dim_out = shape

    param = _WEIGHT_INIT_PARAM[(dist, std)](dim_in, dim_out)

    if dist == "normal":
        return param

    return -param, param
