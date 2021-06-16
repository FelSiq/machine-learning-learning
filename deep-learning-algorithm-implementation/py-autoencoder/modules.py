import numpy as np


class _BaseLayer:
    def __init__(self):
        self._cache = []
        self.trainable = False
        self.frozen = False

    def _store_in_cache(self, *args):
        if self.frozen:
            return

        self._cache.append(args)

    def _pop_from_cache(self):
        return self._cache.pop()

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError


class Linear(_BaseLayer):
    def __init__(self, dim_in: int, dim_out: int):
        super(Linear, self).__init__()

        he_init_coef = np.sqrt(2.0 / dim_in)
        self.weights = he_init_coef * np.random.randn(dim_in, dim_out)
        self.bias = np.zeros(dim_out, dtype=float)

        self.trainable = True

    def forward(self, X):
        out = X @ self.weights + self.bias
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        dW = X.T @ dout
        dX = dout @ self.weights.T
        db = np.sum(dout, axis=0)

        return dX, dW, db

    def update(self, *args):
        if self.frozen:
            return

        dW, db = args
        self.weights -= dW
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
