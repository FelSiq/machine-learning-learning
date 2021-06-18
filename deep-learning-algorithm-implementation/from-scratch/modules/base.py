import typing as t

import numpy as np


class _BaseLayer:
    def __init__(self, trainable: bool = False):
        self._cache = []
        self.trainable = trainable
        self.frozen = False
        self.parameters = tuple()
        self.layers = tuple()

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

    def clean_grad_cache(self):
        self._cache.clear()

    @property
    def has_stored_grads(self):
        return len(self._cache) > 0 or any(l.has_stored_grads for l in self.layers)

    def train(self):
        self.frozen = False
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.frozen = True
        for layer in self.layers:
            layer.eval()


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
        out = X @ self.weights

        if self.bias.size:
            out += self.bias

        self._store_in_cache(X)

        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        dW = X.T @ dout
        dX = dout @ self.weights.T

        if self.bias.size:
            db = np.sum(dout, axis=0)
            return (dX,), (dW, db)

        return (dX,), (dW,)

    def update(self, *args):
        if self.frozen:
            return

        if self.bias.size:
            (dW, db) = args
            self.bias -= db
            return

        (dW,) = args
        self.weights -= dW


class Reshape(_BaseLayer):
    def __init__(self, out_shape: t.Tuple[int, ...]):
        assert len(out_shape)
        super(Flatten, self).__init__()
        self.out_shape = tuple(out_shape)

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.reshape(self.out_shape)

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        dout = dout.reshape(shape)
        return dout


class Flatten(Reshape):
    def __init__(self):
        super(Flatten, self).__init__(out_shape=(-1,))

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.ravel()
