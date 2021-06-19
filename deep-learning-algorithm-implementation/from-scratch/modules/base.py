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
        assert dim_in > 0
        assert dim_out > 0

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


class MultiLinear(_BaseLayer):
    def __init__(self, dims_in: t.Tuple[int], dim_out: int, add_bias: bool = True):
        self.layers = [
            Linear(dim_in, dim_out, add_bias=False) for dim_in in dims_in[:-1]
        ]
        self.layers.append(Linear(dims_in[-1], dim_out, add_bias=add_bias))
        self.layers = tuple(self.layer)

        self.dim_out = int(dim_out)

        self.parameters = []

        for layer in self.layers:
            self.parameters.extend(layer.parameters)

        self.parameters = tuple(self.parameters)

    def forward(self, X):
        out = np.zeros((X.shape[0], self.dim_out), dtype=float)

        for layer in self.layers:
            out += layer(out)

        return out

    def backward(self, dout):
        d_outs = []  # type: t.List[np.ndarray]
        d_params = []  # type: t.List[np.ndarray]

        for layer in self.layers:
            grads = layer.backward(dout)
            d_outs.extend(grads[0])
            d_params.extend(grads[1])

        return tuple(d_outs), tuple(d_params)

    def update(self, *args):
        if self.frozen:
            return

        for layer, d_params in zip(self.layers, args):
            layer.update(d_params)


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


class WeightedAverage(_BaseLayer):
    def __call__(self, X, Y, W):
        return self.forward(X, Y, W)

    def forward(self, X, Y, W):
        out = W * X + (1.0 - W) * Y
        self._store_in_cache(X, Y, W)
        return out

    def backward(self, dout):
        X, Y, W = self._pop_from_cache()
        dX = W * dout
        dY = (1.0 - W) * dout
        dW = (X - Y) * dout
        return dX, dY, dW
