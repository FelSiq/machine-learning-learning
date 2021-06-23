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


def weight_init_param_he(dist: str, dim_in: int, *args):
    assert dist in {"normal", "uniform"}

    if dist == "normal":
        return np.sqrt(2.0 / dim_in)

    return np.sqrt(6.0 / dim_in)


def weight_init_param_xavier(dist: str, dim_in: int, *args):
    assert dist in {"normal", "uniform"}

    if dist == "normal":
        return np.sqrt(1.0 / (3.0 * dim_in))

    return np.sqrt(1.0 / dim_in)


def weight_init_param_xavier_norm(dist: str, dim_in: int, dim_out: int, *args):
    assert dist in {"normal", "uniform"}

    if dist == "normal":
        return np.sqrt(2.0 / (dim_in + dim_out))

    return np.sqrt(6.0 / (dim_in + dim_out))


# NOTE: either 'normal' and 'uniform' distributions are initialized
# to have the very same variance.
_WEIGHT_INIT_PARAM = {
    "he": weight_init_param_he,
    "xavier": weight_init_param_xavier,
    "xavier_norm": weight_init_param_xavier_norm,
}

# Known rules of thumb:
# 'he': suitable for ReLU activations
# 'xavier': suitable for Tanh and Sigmoid activations


def get_weight_init_dist_params(
    std: t.Union[str, float],
    dist: str,
    shape: t.Union[int, int],
    dims: t.Optional[t.Tuple[int, int]] = None,
):
    if np.isreal(std):
        assert dist == "normal"
        return float(std)

    assert dist in {"normal", "uniform"}
    assert not isinstance(std, str) or std in {"he", "xavier", "xavier_norm"}

    if dims is not None:
        dim_in, dim_out = dims if hasattr(dims, "__len__") else (dims, dims)

    else:
        dim_in, dim_out = shape

    init_type = std

    param = _WEIGHT_INIT_PARAM[init_type](dist, dim_in, dim_out)

    if dist == "normal":
        return param

    return -param, param
