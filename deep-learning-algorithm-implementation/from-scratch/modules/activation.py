import numpy as np

from . import base


class ReLU(base.BaseLayer):
    def forward(self, X):
        out = np.maximum(X, 0.0)
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        dout = (X > 0.0).astype(float, copy=False) * dout
        return dout


class LeakyReLU(base.BaseLayer):
    def __init__(self, slope: float):
        assert float(slope) >= 0.0
        super(LeakyReLU, self).__init__()
        self.slope = float(slope)

    def forward(self, X):
        out = np.maximum(X, self.slope * X)
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        dout = np.maximum(X > 0.0, self.slope, dtype=float) * dout
        return dout


class Tanh(base.BaseLayer):
    def forward(self, X):
        out = np.tanh(X)
        self._store_in_cache(out)
        return out

    def backward(self, dout):
        (tanh_X,) = self._pop_from_cache()
        dout = (1.0 - np.square(tanh_X)) * dout
        return dout


class Sigmoid(base.BaseLayer):
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
