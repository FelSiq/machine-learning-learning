import numpy as np


def loss_bce(AL, Y):
    m = Y.shape[1]
    loss = -np.sum(np.dot(Y, np.log(AL).T) + np.dot(1.0 - Y, np.log(1.0 - AL).T)) / m
    return np.squeeze(loss)


def loss_bce_grad(AL, Y):
    return -(np.divide(Y, AL) - np.divide(1.0 - Y, 1.0 - AL))


LOSSES = {
    "bce": (loss_bce, loss_bce_grad),
}


def forward(AL, Y, loss_func: str):
    loss_func, loss_func_grad = LOSSES[loss_func]
    loss = loss_func(AL, Y)

    cache = (loss_func_grad, Y)

    return loss, cache


def backward(AL, cache):
    loss_func_grad, Y = cache
    return loss_func_grad(AL, Y.reshape(AL.shape))
