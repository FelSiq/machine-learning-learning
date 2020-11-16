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


def forward_dropout(A_prev, keep_prob: float = 1.0):
    if keep_prob >= 1.0:
        return A_prev, (None, keep_prob)

    mask = (np.random.rand(*A_prev.shape) <= keep_prob).astype(float)

    A = a_prev * mask
    A /= keep_prob

    cache = (mask, keep_prob)

    return A, cache


def backward_dropout(dA, cache):
    mask, keep_prob = cache

    if keep_prob >= 1.0:
        return dA

    dA_prev = mask * dA / keep_prob

    return dA_prev
