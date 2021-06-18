import numpy as np

from . import base


class ReLU(base._BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, X):
        out = np.maximum(X, 0.0)
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        dout = (X > 0.0).astype(float, copy=False) * dout
        return dout


class Tanh(base._BaseLayer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, X):
        out = np.tanh(X)
        self._store_in_cache(out)
        return out

    def backward(self, dout):
        (tanh_X,) = self._pop_from_cache()
        dout = (1.0 - np.square(tanh_X)) * dout
        return dout


class Sigmoid(base._BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, X):
        inds_pos = X >= 0
        inds_neg = ~inds_pos

        exp_neg = np.exp(X[inds_neg])

        out = np.zeros_like(X, dtype=float)
        out[inds_pos] = 1.0 / (1.0 + np.exp(-X[inds_pos]))
        out[inds_neg] = exp_neg / (1.0 + exp_neg)

        self._store_in_cache(out)

        return out

    def backward(self, dout):
        (sig_X,) = self._pop_from_cache()
        dout = sig_X * (1.0 - sig_X) * dout
        return dout
