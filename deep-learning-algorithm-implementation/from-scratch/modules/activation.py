import numpy as np

from . import base


class ReLU(base.BaseLayer):
    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = bool(inplace)

    def forward(self, X):
        out = X
        out = np.maximum(out, 0.0, out=out if self.inplace else None)
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        dout = (X > 0.0).astype(float, copy=False) * dout
        return dout


class LeakyReLU(base.BaseLayer):
    def __init__(self, slope: float, inplace: bool = False):
        assert float(slope) >= 0.0

        super(LeakyReLU, self).__init__()

        self.slope = float(slope)
        self.inplace = bool(inplace)

    def forward(self, X):
        out = X
        out = np.maximum(out, self.slope * out, out=out if self.inplace else None)
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        dout_b = (X > 0.0).astype(float, copy=False)
        dout_b = np.maximum(dout_b, self.slope, out=dout_b)
        dout_b *= dout
        return dout_b


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


class Softmax(base.BaseLayer):
    def __init__(self):
        super(Softmax, self).__init__()

        raise NotImplementedError

        self.exp = base.Exp()
        self.sum = base.Sum(axis=-1)
        self.divide = base.Divide()

        self.register_layers(self.exp, self.sum, self.divide)

    def forward(self, X):
        exp = self.exp(X - np.max(X, axis=-1, keepdims=True))
        sum_exp = self.sum(exp)
        probs = self.divide(exp, sum_exp)
        return probs

    def backward(self, dout):
        d_exp_a, d_sum_exp = self.divide.backward(dout)
        d_exp_b = self.sum.backward(d_sum_exp)
        d_exp = d_exp_a + d_exp_b
        dX = self.exp.backward(d_exp)
        return dX
