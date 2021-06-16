import typing as t

import numpy as np


class _BaseLayer:
    def __init__(self, trainable: bool = False):
        self._cache = []
        self.trainable = trainable
        self.frozen = False
        self.parameters = tuple()

    def _store_in_cache(self, *args):
        if self.frozen:
            return

        self._cache.append(args)

    def _pop_from_cache(self):
        return self._cache.pop()

    def forward(self, X):
        raise NotImplementedError

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dout):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError

    @property
    def has_stored_grads(self):
        return len(self._cache) > 0


class Linear(_BaseLayer):
    def __init__(self, dim_in: int, dim_out: int, include_bias: bool = True):
        super(Linear, self).__init__(trainable=True)

        he_init_coef = np.sqrt(2.0 / dim_in)
        self.weights = he_init_coef * np.random.randn(dim_in, dim_out)
        self.bias = np.empty(0, dtype=float)

        if include_bias:
            self.bias = np.zeros(dim_out, dtype=float)
            self.parameters = (self.weights, self.bias)

        else:
            self.parameters = (self.weights,)

    def forward(self, X):
        out = X @ self.weights + self.bias
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        dW = X.T @ dout
        dX = dout @ self.weights.T

        db = np.empty(0, dtype=float)
        if self.bias.size:
            db = np.sum(dout, axis=0)

        return dX, dW, db

    def update(self, *args):
        if self.frozen:
            return

        dW, db = args
        self.weights -= dW

        if self.bias.size:
            self.bias -= db


class ReLU(_BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, X):
        out = np.maximum(X, 0.0)
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        return (X > 0.0) * dout


class Tanh(_BaseLayer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, X):
        out = np.tanh(X)
        self._store_in_cache(out)
        return out

    def backward(self, dout):
        (tanh_X,) = self._pop_from_cache()
        return (1.0 - np.square(tanh_X)) * dout


class Sigmoid(_BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, X):
        inds_pos = x >= 0
        inds_neg = ~inds_pos

        exp_neg = np.exp(x[inds_neg])

        out = np.zeros_like(x, dtype=float)
        out[inds_pos] = 1.0 / (1.0 + np.exp(-x[inds_pos]))
        out[inds_neg] = exp_neg / (1.0 + exp_neg)

        self._store_in_cache(out)

        return out

    def backward(self, dout):
        (sig_X,) = self._pop_from_cache()
        return sig_X * (1.0 - sig_X)


class Reshape(_BaseLayer):
    def __init__(self, out_shape: t.Tuple[int, ...]):
        assert len(out_shape)
        super(Flatten, self).__init__()
        self.out_shape = tuple(out_shape)

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.reshape()

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        return dout.reshape(shape)


class Flatten(Reshape):
    def __init__(self):
        super(Flatten, self).__init__(out_shape=(-1,))

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.ravel()
