"""Forward and Backward methods."""
import numpy as np

import activations
import losses


def forward_linear(A, W, b):
    Z = W @ A + b

    cache = (A, W, Z)

    return Z, cache


def backward_linear(dZ, cache, lambd: float = 0.0):
    A_prev, W, Z = cache

    m = A_prev.shape[1]

    dW = dZ @ A_prev.T / m

    if lambd > 0.0:
        dW += lambd * W

    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def forward_linear_activation(A_prev, W, b, activation: str):
    Z, cache_linear = forward_linear(A_prev, W, b)
    A, cache_activation = activations.forward(Z, activation)

    cache = (cache_activation, cache_linear)

    return A, cache


def backward_linear_activation(dA, cache, lambd: float = 0.0):
    cache_activation, cache_linear = cache

    dZ = activations.backward(dA, cache_activation)
    dA_prev, dW, db = backward_linear(dZ, cache_linear, lambd=lambd)

    return dA_prev, dW, db
