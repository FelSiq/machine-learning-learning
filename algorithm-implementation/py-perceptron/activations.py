import numpy as np


def sigmoid(x):
    inds_pos = x >= 0.0
    inds_neg = ~inds_pos

    exp_neg = np.exp(x[inds_neg])

    res = np.zeros_like(x, dtype=float)
    res[inds_pos] = 1.0 / (1.0 + np.exp(-x[inds_pos]))
    res[inds_neg] = exp_neg / (1.0 + exp_neg)

    return res

def sigmoid_grad(y_pred):
    return y_pred * (1.0 - y_pred)


def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(y_pred):
    return np.heaviside(y_pred, 1.0)


def tanh(x):
    return np.tanh(x)


def tanh_grad(y_pred):
    return 1.0 - np.square(y_pred)


_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
}

def get_activation(activation):
    return _ACTIVATIONS[activation]
