import numpy as np

from . import base
from . import activation


class _BaseSequenceCell(base._BaseLayer):
    def __init__(self, dim_in: int, dim_hidden: int):
        assert int(dim_in) > 0
        assert int(dim_hidden) > 0

        super(_BaseSequenceCell, self).__init__(trainable=True)

        self.dim_in = int(dim_in)
        self.dim_hidden = int(dim_hidden)

        self.reset()

    def reset(self):
        self.cell_state = np.empty(0, dtype=float)

    def _prepare_cell_state(self, X):
        if self.cell_state.size == 0:
            batch_size = X.shape[0]
            self.cell_state = np.zeros((batch_size, self.dim_hidden), dtype=float)


class RNNCell(_BaseSequenceCell):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(RNNCell, self).__init__(dim_in, dim_hidden)

        self.lin_hidden = base.Linear(dim_hidden, dim_hidden, include_bias=False)
        self.lin_input = base.Linear(dim_in, dim_hidden, include_bias=True)
        self.tanh_layer = activation.Tanh()

        self.layers = (
            self.lin_hidden,
            self.lin_input,
            self.tanh_layer,
        )

        self.parameters = (
            *self.lin_hidden.parameters,
            *self.lin_input.parameters,
        )

    def forward(self, X):
        self._prepare_cell_state(X)
        aux_cell_state = self.lin_hidden(self.cell_state)
        aux_X = self.lin_input(X)
        self.cell_state = self.tanh_layer(aux_cell_state + aux_X)
        return self.cell_state

    def backward(self, dout_h, dout_y):
        dout = self.tanh_layer.backward(dout_h + dout_y)
        ((dh,), (dWh,)) = self.lin_hidden.backward(dout)
        ((dX,), (dWX, dbX)) = self.lin_input.backward(dout)
        return (dh, dX), (dWh, dWX, dbX)

    def update(self, *args):
        if self.frozen:
            return

        (dWh, dWX, dbX) = args

        self.lin_hidden.update(dWh)
        self.lin_input.update(dWX, dbX)


class Embedding(base._BaseLayer):
    def __init__(self, num_tokens: int, dim_embedding: int):
        assert int(num_tokens) > 0
        assert int(dim_embedding) > 0

        super(Embedding, self).__init__(trainable=True)

        self.embedding = np.random.random((num_tokens, dim_embedding))
        self.parameters = (self.embedding,)

    def forward(self, X):
        embedded_tokens = self.embedding[X, :]
        self._store_in_cache(X)
        return embedded_tokens

    def backward(self, dout):
        (orig_inds,) = self._pop_from_cache()
        dout = self.embedding[orig_inds, :] * dout
        dout_b = np.zeros_like(self.embedding)
        np.add.at(dout_b, orig_inds, dout)
        return tuple(), (dout_b,)

    def update(self, *args):
        if self.frozen:
            return

        (demb,) = args
        self.embedding -= demb
