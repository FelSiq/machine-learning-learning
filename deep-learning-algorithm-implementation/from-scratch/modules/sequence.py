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

        self.lin_hidden_z = base.Linear(dim_hidden, dim_hidden, include_bias=False)
        self.lin_input_z = base.Linear(dim_in, dim_hidden, include_bias=True)

        self.lin_hidden_r = base.Linear(dim_hidden, dim_hidden, include_bias=False)
        self.lin_input_r = base.Linear(dim_in, dim_hidden, include_bias=True)

        self.lin_hidden_h = base.Linear(dim_hidden, dim_hidden, include_bias=False)
        self.lin_input_h = base.Linear(dim_in, dim_hidden, include_bias=True)

        self.weighted_avg = base.WeightedAverage()

        self.tanh_layer = activation.Tanh()
        self.sigm_layer = activation.Sigmoid()

        self.layers = (
            self.lin_hidden_z,
            self.lin_input_z,
            self.lin_hidden_r,
            self.lin_input_r,
            self.lin_hidden_h,
            self.lin_input_h,
            self.weighted_avg,
            self.tanh_layer,
            self.sigm_layer,
        )

        self.parameters = (
            *self.lin_hidden_z.parameters,
            *self.lin_input_z.parameters,
            *self.lin_hidden_r.parameters,
            *self.lin_input_r.parameters,
            *self.lin_hidden_h.parameters,
            *self.lin_input_h.parameters,
        )

    def forward(self, X):
        self._prepare_cell_state(X)

        z = self.lin_hidden_z(self.cell_state) + self.lin_input_z(X)
        z = self.sigm_layer(z)

        r = self.lin_hidden_r(self.cell_state) + self.lin_input_r(X)
        r = self.sigm_layer(r)

        r_cs = r * self.cell_state

        h = self.lin_hidden_h(r_cs) + self.lin_input_h(X)
        h = self.tanh_layer(h)

        self._store_in_cache(self.cell_state, r)

        self.cell_state = self.weighted_avg(self.cell_state, h, z)

        return self.cell_state

    def backward(self, dout):
        (prev_cell_state, r) = self._pop_from_cache()

        (d_cell_state, dh, dz) = self.weighted_avg.backward(dout)

        dh = self.tanh_layer.backward(dh)
        ((dr_cs,), (dWhh,)) = self.lin_hidden_h.backward(dh)
        ((dXh,), (dWih, dbih)) = self.lin_input_h.backward(dh)

        dr = prev_cell_state * dr_cs
        dhh = r * dr_cs

        dr = self.sigm_layer.backward(dr)
        ((dhr,), (dWhr,)) = self.lin_hidden_r.backward(dr)
        ((dXr,), (dWir, dbir)) = self.lin_input_r.backward(dr)

        dz = self.sigm_layer.backward(dr)
        ((dhz,), (dWhz,)) = self.lin_hidden_z.backward(dz)
        ((dXz,), (dWiz, dbiz)) = self.lin_input_z.backward(dz)

        dX = dXz + dXr + dXh
        dh = dhz + dhr + dhh

        return ((dh, dX), (dWhh, dWih, dbih, dWhr, dWir, dbir, dWhz, dWiz, dbiz))

    def update(self, *args):
        if self.frozen:
            return

        (dWhh, dWih, dbih, dWhr, dWir, dbir, dWhz, dWiz, dbiz) = args

        self.lin_hidden_h.update(dWhh)
        self.lin_hidden_r.update(dWhr)
        self.lin_hidden_z.update(dWhz)
        self.lin_input_h.update(dWih, dbih)
        self.lin_input_r.update(dWir, dbir)
        self.lin_input_z.update(dWiz, dbiz)


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
