import typing as t

import numpy as np


def calc_init_val(
    layer_dims: t.Sequence[int], cur_layer: int, init_val: t.Union[float, str]
) -> float:
    if isinstance(init_val, (int, float)):
        return float(init_val)

    if init_val == "xavier":
        # Note: recommended for Tanh activations
        return np.sqrt(1.0 / layer_dims[cur_layer - 1])

    if init_val == "he":
        # Note: recommended for ReLU activations
        return np.sqrt(2.0 / layer_dims[cur_layer - 1])

    raise ValueError(f"Unknown initilization: {init_val}.")


def initialize_parameters(
    layer_dims: t.Sequence[int],
    init_val: t.Union[str, float] = "he",
    random_state: t.Optional[int] = None,
    make_assertions: bool = True,
) -> t.Dict[str, np.ndarray]:

    if make_assertions:
        assert (
            not isinstance(init_val, float) or init_val > 0.0
        ), "'init_val' must be positive to ensure assymetry in network."
        assert len(layer_dims) >= 2, "Network must have at least 2 layers"

    if random_state is not None:
        np.random.seed(random_state)

    parameters = dict()

    for l in np.arange(1, len(layer_dims)):
        weight_std = calc_init_val(layer_dims, l, init_val)

        parameters["W" + str(l)] = weight_std * np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        )

        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def _test():
    layer_dims = [2, 4, 2]

    print("Layer dimensions:", layer_dims)

    params = initialize_parameters(layer_dims, "xavier")
    params = initialize_parameters(layer_dims, "he")
    params = initialize_parameters(layer_dims, 0.01)

    for k, v in params.items():
        print(k, v.shape)

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    X = np.random.random((2, 10))
    z1 = W1 @ X + b1
    z2 = W2 @ z1 + b2

    assert z2.shape == (2, 10)


if __name__ == "__main__":
    _test()
