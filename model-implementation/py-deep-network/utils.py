import typing as t

import numpy as np


def initialize_parameters(
    layer_dims: t.Sequence[int],
    random_state: t.Optional[int] = None,
    weight_std: float = 0.01,
    make_assertions: bool = True,
) -> t.Dict[str, np.ndarray]:

    if make_assertions:
        assert (
            weight_std > 0.0
        ), "'weight_std' must be positive to ensure assymetry in network."
        assert len(layer_dims) >= 2, "Network must have at least 2 layers"

    if random_state is not None:
        np.random.seed(random_state)

    parameters = dict()

    for l in np.arange(1, len(layer_dims)):
        parameters["W" + str(l)] = weight_std * np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        )
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def _test():
    layer_dims = [2, 4, 2]

    print("Layer dimensions:", layer_dims)

    params = initialize_weights(layer_dims)

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
