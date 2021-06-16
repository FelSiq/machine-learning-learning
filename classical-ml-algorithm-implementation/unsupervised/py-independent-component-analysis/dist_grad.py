import numpy as np


def sigmoid(x):
    inds_pos = x >= 0
    inds_neg = ~inds_pos

    exp_neg = np.exp(x[inds_neg])

    res = np.zeros_like(x, dtype=float)
    res[inds_pos] = 1.0 / (1.0 + np.exp(-x[inds_pos]))
    res[inds_neg] = exp_neg / (1.0 + exp_neg)

    return res


def sigmoid_grad(x, *args):
    return 1.0 - 2.0 * sigmoid(x)


def laplace_grad(x, mu=0.0, b=1.0, *args):
    return np.sign(x - mu) / -b


_DIST_GRADS = {
    "sigmoid": sigmoid_grad,
    "laplace": laplace_grad,
}


def get_dist_grad_fun(dist):
    return _DIST_GRADS[dist]
