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
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super(Linear, self).__init__()

        he_init_coef = np.sqrt(2.0 / dim_in)
        self.weights = he_init_coef * np.random.randn(dim_in, dim_out)

        self.bias = np.empty(0, dtype=float)
        if bias:
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

        db = np.empty(0, dtype=float)

        if self.bias.size:
            db = np.sum(dout, axis=0)

        return dX, dW, db

    def update(self, *args):
        if self.frozen:
            return

        dW, db = args
        self.weights -= dW
        self.bias -= db


class RNNCell(_BaseLayer):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super(RNNCell, self).__init__()

        assert dim_in > 0
        assert dim_hidden > 0
        assert dim_out > 0

        self.lin_hidden = Linear(dim_hidden, dim_hidden, bias=False)
        self.lin_input = Linear(dim_in, dim_hidden, bias=True)

        self.cell_state = np.zeros(dim_hidden, dtype=float)

    def forward(self, X):
        aux_cell_state = self.lin_hidden(self.cell_state)
        aux_X = self.lin_input(X)

        self.cell_state = np.tanh(aux_cell_state + aux_X)

        self._store_in_cache(self.cell_state)

        return self.cell_state

    def backward(self, dout_h, dout_y):
        (cell_state,) = self._pop_from_cache()

        dcs = (1.0 - np.square(cell_state)) * (dout_h + dout_y)

        (dh, dWh, dbh) = self.lin_hidden.backward(dcs)
        (dX, dWX, dbX) = self.lin_input.backward(dcs)

        return (dh, dX, dWh, dbh, dWX, dbX)

    def update(self, *args):
        if self.frozen:
            return

        (dWh, dbh, dWX, dbX) = args

        self.lin_hidden.update(dWh, dbh)
        self.lin_input.update(dWX, dbX)
