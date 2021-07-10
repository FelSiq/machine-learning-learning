import numpy as np
import scipy.special
import scipy.stats

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


class LogSoftmax(base.BaseLayer):
    def __init__(self, axis: int = -1):
        super(LogSoftmax, self).__init__()

        self.exp = base.Exp()
        self.sum = base.Sum(axis=axis)
        self.log = base.Log()
        self.sub = base.Subtract()

        self.axis = self.sum.axis

        self.register_layers(self.exp, self.sum, self.log, self.sub)

    def forward(self, X):
        X = X - np.max(X, axis=self.axis, keepdims=True)
        X_lse = self.log(self.sum(self.exp(X)))
        out = self.sub(X, X_lse)
        return out

    def backward(self, dout):
        dX_a, dX_lse = self.sub.backward(dout)
        dX_b = self.exp.backward(self.sum.backward(self.log.backward(dX_lse)))
        dX = dX_a + dX_b
        return dX


class Softmax(base.BaseLayer):
    def __init__(self, axis: int = -1):
        super(Softmax, self).__init__()

        self.log_softmax = LogSoftmax(axis=axis)
        self.exp = base.Exp()

        self.register_layers(self.log_softmax, self.exp)

    def forward(self, X):
        X_logits = self.log_softmax(X)
        out = self.exp(X_logits)
        return out

    def backward(self, dout):
        dX_logits = self.exp.backward(dout)
        dX = self.log_softmax.backward(dX_logits)
        return dX


class GELU(base.BaseLayer):
    def __init__(self):
        super(GELU, self).__init__()
        self.mult = base.Multiply()
        self.mult_const_ca = base.ScaleByConstant(0.5)
        self.mult_const_cb = base.ScaleByConstant(np.sqrt(0.5))
        self.register_layers(self.mult, self.mult_const_ca, self.mult_const_cb)
        self._norm_pdf = scipy.stats.norm(loc=0.0, scale=np.sqrt(0.5)).pdf

    def forward(self, X):
        X_ca = self.mult_const_ca(X)
        X_cb = self.mult_const_cb(X)
        X_erf = 1.0 + scipy.special.erf(X_cb)
        out = self.mult(X_ca, X_erf)
        self._store_in_cache(X_cb)
        return out

    def backward(self, dout):
        (X_cb,) = self._pop_from_cache()

        dX_ca, dX_erf = self.mult.backward(dout)
        dX_cb = 2.0 * self._norm_pdf(X_cb) * dX_erf

        dX_b = self.mult_const_cb.backward(dX_cb)
        dX_a = self.mult_const_ca.backward(dX_ca)

        dX = dX_a + dX_b

        return dX


class CosineSimilarity(base.BaseLayer):
    def __init__(self, pairwise: bool = True):
        super(CosineSimilarity, self).__init__()

        self.rowwise = not bool(pairwise)

        if self.rowwise:
            self.mult = base.Multiply()
            self.sum = base.Sum(axis=1)
            self.register_layers(self.mult, self.sum)

        else:
            self.mult = base.Matmul(transpose_X=True)
            self.register_layers(self.mult)

        self.norm_vec = base.NormalizeVector(p=2, axis=1)
        self.register_layers(self.norm_vec)

    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        X_norm = self.norm_vec(X)
        Y_norm = self.norm_vec(Y)
        out = self.mult(X_norm, Y_norm)

        if self.rowwise:
            out = self.sum(out)

        return out

    def backward(self, dout):
        if self.rowwise:
            dout = self.sum.backward(dout)

        dX_norm, dY_norm = self.mult.backward(dout)

        dY = self.norm_vec.backward(dY_norm)
        dX = self.norm_vec.backward(dX_norm)

        return dX, dY
