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


def forward_batchnorm(
    Z: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    moving_avg: np.ndarray,
    moving_std: np.ndarray,
    momentum: float = 0.9,
    epsilon: float = 1e-6,
    test_time: bool = False,
):
    batch_avg = np.mean(Z, keepdims=True, axis=1)
    batch_std = np.std(Z, keepdims=True, axis=1)

    if test_time:
        Z_norm = (Z - moving_avg) / (moving_std + epsilon)

    else:
        Z_norm = (Z - batch_avg) / (batch_std + epsilon)

    Z_scaled = G * Z_norm + B

    cache = (
        G,
        B,
        Z_norm,
        momentum,
        batch_avg,
        batch_std,
        moving_avg,
        moving_std,
        epsilon,
    )

    return Z_scaled, cache


def backward_batchnorm(dZ_scaled, cache):
    (
        G,
        B,
        Z_norm,
        momentum,
        batch_avg,
        batch_std,
        moving_avg,
        moving_std,
        epsilon,
    ) = cache

    m = Z_norm.shape[1]

    dZ_norm = dZ_scaled * G
    dG = np.sum(dZ_scaled * Z_norm, axis=1, keepdims=True) / m
    dB = np.sum(dZ_scaled, axis=1, keepdims=True) / m
    dZ = dZ_norm / (batch_std + epsilon)

    moving_avg = momentum * moving_avg + (1. - momentum) * batch_avg
    moving_std = momentum * moving_std + (1. - momentum) * batch_std

    return dZ_norm, dG, dB, moving_avg, moving_std


def forward_linear_batchnorm_activation(
    A_prev,
    W,
    b,
    G,
    B,
    moving_avg,
    moving_std,
    activation: str,
    momentum: float = 0.9,
    bias_correction: bool = True,
    epsilon: float = 1e-6,
    test_time: bool = False,
):
    Z, cache_linear = forward_linear(A_prev, W, b)
    Z_scaled, cache_batchnorm = forward_batchnorm(Z, G, B, moving_avg, moving_std, test_time=test_time)
    A, cache_activation = activations.forward(Z_scaled, activation)

    cache = (cache_activation, cache_batchnorm, cache_linear)

    return A, cache


def backward_linear_batchnorm_activation(dA, cache, lambd: float = 0.0):
    cache_activation, cache_batchnorm, cache_linear = cache

    dZ_scaled = activations.backward(dA, cache_activation)
    dZ, dG, dB, moving_avg, moving_std = backward_batchnorm(
        dZ_scaled, cache_batchnorm
    )
    dA_prev, dW, db = backward_linear(dZ, cache_linear, lambd=lambd)

    return dA_prev, dW, db, dG, dB, moving_avg, moving_std
