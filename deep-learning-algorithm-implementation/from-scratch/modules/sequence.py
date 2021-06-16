import numpy as np

from . import base


class Embedding(base._BaseLayer):
    def __init__(self, num_tokens: int, dim_embedding: int):
        assert int(num_tokens) > 0
        assert int(dim_embedding) > 0

        super(Embedding, self).__init__()

        # TODO: how to initialize an embedding layer properly?
        self.embedding = np.random.randn(num_tokens, dim_embedding)

        self.trainable = True

    def forward(self, X):
        embedded_tokens = self.embedding[X, :]
        self._store_in_cache(X, X)
        return embedded_tokens

    def backward(self, dout):
        (orig_inds,) = self._pop_from_cache()
        return self.embedding[orig_inds, :] * dout

    def update(self, *args):
        (orig_inds,) = self._pop_from_cache()

        if self.frozen:
            return

        (demb,) = args
        np.subtract.at(self.embedding, orig_inds, demb)


class RNNCell(base._BaseLayer):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        assert int(dim_in) > 0
        assert int(dim_hidden) > 0
        assert int(dim_out) > 0

        super(RNNCell, self).__init__()

        self.lin_hidden = base.Linear(dim_hidden, dim_hidden, include_bias=False)
        self.lin_input = base.Linear(dim_in, dim_hidden, include_bias=True)
        self.tanh_layer = base.Tanh()

        self.cell_state = np.zeros(dim_hidden, dtype=float)

        self.trainable = True

    def forward(self, X):
        aux_cell_state = self.lin_hidden(self.cell_state)
        aux_X = self.lin_input(X)
        self.cell_state = self.tanh_layer(aux_cell_state + aux_X)
        return self.cell_state

    def backward(self, dout_h, dout_y):
        dout = self.tanh_layer.backward(dout_h + dout_y)
        (dh, dWh, dbh) = self.lin_hidden.backward(dout)
        (dX, dWX, dbX) = self.lin_input.backward(dout)
        return (dh, dX, dWh, dbh, dWX, dbX)

    def update(self, *args):
        if self.frozen:
            return

        (dWh, dbh, dWX, dbX) = args

        self.lin_hidden.update(dWh, dbh)
        self.lin_input.update(dWX, dbX)
