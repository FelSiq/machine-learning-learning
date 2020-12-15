import typing as t

import numpy as np


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def grad_sigmoid(A):
    return A * (1.0 - A)


def grad_tanh(A):
    return 1.0 - np.square(A)


def relu(Z):
    return np.maximum(Z, 0.0)


def grad_relu(A):
    return (A > 0.0).astype(float)


def identity(Z):
    return Z


def grad_identity(Z):
    return 1.0


ACTIVATIONS = {
    "sigmoid": (sigmoid, grad_sigmoid),
    "logistic": (sigmoid, grad_sigmoid),
    "tanh": (np.tanh, grad_tanh),
    "ReLU": (relu, grad_relu),
    "relu": (relu, grad_relu),
    "identity": (identity, grad_identity),
}


def forward(Z, activation: str):
    activation_func, activation_grad = ACTIVATIONS[activation]
    A = activation_func(Z)
    cache = (A, activation_grad)
    return A, cache


def backward(dA, cache):
    A, activation_grad = cache
    return activation_grad(A) * dA
