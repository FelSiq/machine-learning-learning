import numpy as np

from . import base
from . import activation


class _BaseSequenceCell(base.BaseLayer):
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
        aux_cell_state = self.lin_hidden(self.cell_state) + self.lin_input(X)
        self.cell_state = self.tanh_layer(aux_cell_state)
        return self.cell_state

    def backward(self, dout):
        dout = self.tanh_layer.backward(dout)

        ((dh,), (dWh,)) = self.lin_hidden.backward(dout)
        ((dX,), (dWX, dbX)) = self.lin_input.backward(dout)

        return (dh, dX), (dWh, dWX, dbX)

    def update(self, *args):
        if self.frozen:
            return

        (dWh, dWX, dbX) = args

        self.lin_hidden.update(dWh)
        self.lin_input.update(dWX, dbX)


class GRUCell(_BaseSequenceCell):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(GRUCell, self).__init__(dim_in, dim_hidden)

        self.lin_z = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Sigmoid()
        )
        self.lin_r = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Sigmoid()
        )
        self.lin_h = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Tanh()
        )

        self.weighted_avg = base.WeightedAverage()

        self.layers = (
            self.lin_z,
            self.lin_r,
            self.lin_h,
            self.weighted_avg,
        )

        self.parameters = (
            *self.lin_z.parameters,
            *self.lin_r.parameters,
            *self.lin_h.parameters,
        )

    def forward(self, X):
        self._prepare_cell_state(X)

        z = self.lin_z(self.cell_state, X)
        r = self.lin_r(self.cell_state, X)

        r_cs = r * self.cell_state

        h = self.lin_h(r_cs, X)

        self._store_in_cache(self.cell_state, r)

        self.cell_state = self.weighted_avg(self.cell_state, h, z)

        return self.cell_state

    def backward(self, dout):
        (prev_cell_state, r) = self._pop_from_cache()

        (d_cell_state, dh, dz) = self.weighted_avg.backward(dout)

        ((dr_cs, dXh), lin_h_params) = self.lin_h.backward(dh)

        dr = prev_cell_state * dr_cs
        dhh = r * dr_cs

        ((dhr, dXr), lin_r_params) = self.lin_r.backward(dr)
        ((dhz, dXz), lin_z_params) = self.lin_z.backward(dz)

        dh = dhz + dhr + dhh
        dX = dXz + dXr + dXh

        return ((dh, dX), (*lin_z_params, *lin_r_params, *lin_h_params))

    def update(self, *args):
        if self.frozen:
            return

        (dWhz, dWiz, dbiz, dWhr, dWir, dbir, dWhh, dWih, dbih) = args

        self.lin_z.update(dWhz, dWiz, dbiz)
        self.lin_r.update(dWhr, dWir, dbir)
        self.lin_h.update(dWhh, dWih, dbih)


class LSTMCell(_BaseSequenceCell):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(LSTMCell, self).__init__(dim_in, dim_hidden)
        # TODO.

    def forward(self, X):
        self._prepare_cell_state(X)
        # TODO.
        return self.cell_state

    def backward(self, dout):
        # TODO.
        return tuple(), tuple()

    def update(self, *args):
        if self.frozen:
            return
        # TODO.


class Embedding(base.BaseLayer):
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
