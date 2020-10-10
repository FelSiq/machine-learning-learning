"""Forward and Backward methods."""
import numpy as np

import activations
import losses


def forward_linear(A, W, b):
    Z = W @ A + b

    cache = (A, W, Z)

    return Z, cache


def backward_linear(dZ, cache):
    A_prev, W, Z = cache

    m = A_prev.shape[1]

    dW = dZ @ A_prev.T / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def forward_linear_activation(A_prev, W, b, activation: str):
    Z, cache_linear = forward_linear(A_prev, W, b)
    A, cache_activation = activations.forward(Z, activation)

    cache = (cache_activation, cache_linear)

    return A, cache


def backward_linear_activation(dA, cache):
    cache_activation, cache_linear = cache

    dZ = activations.backward(dA, cache_activation)
    dA_prev, dW, db = backward_linear(dZ, cache_linear)

    return dA_prev, dW, db


def forward(
    X, parameters, activation_hidden: str = "ReLU", activation_out: str = "sigmoid",
):
    caches = []
    A = X
    L = len(parameters) // 2

    # Note: hidden layers forward
    for l in np.arange(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        A, cache = forward_linear_activation(
            A_prev=A, W=W, b=b, activation=activation_hidden
        )

        caches.append(cache)

    # Note: output layer forward
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]

    A, cache = forward_linear_activation(A_prev=A, W=W, b=b, activation=activation_out)

    caches.append(cache)

    return A, caches


def backward(
    AL, caches,
):
    grads = dict()
    L = len(caches) - 1
    m = AL.shape[1]

    # Loss gradient
    cur_cache = caches.pop()
    dA = losses.backward(AL, cur_cache)

    # Output layer gradient
    cur_cache = caches.pop()
    dA, dW, db = backward_linear_activation(dA, cur_cache)
    grads["dA" + str(L - 1)] = dA
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # Hidden layers gradient
    for l in np.arange(L - 2, -1, -1):
        cur_cache = caches.pop()
        dA, dW, db = backward_linear_activation(dA, cur_cache)
        grads["dA" + str(l)] = dA
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate: float):
    assert learning_rate > 0.0

    L = len(parameters) // 2

    for l in np.arange(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters


def _test():
    pass


if __name__ == "__main__":
    _test()
