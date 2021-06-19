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
        self.hidden_state = np.empty(0, dtype=float)
        self.cell_state = np.empty(0, dtype=float)

    def _prepare_hidden_state(self, X):
        if self.hidden_state.size == 0:
            batch_size = X.shape[0]
            self.hidden_state = np.zeros((batch_size, self.dim_hidden), dtype=float)


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
        self._prepare_hidden_state(X)
        aux_hidden_state = self.lin_hidden(self.hidden_state) + self.lin_input(X)
        self.hidden_state = self.tanh_layer(aux_hidden_state)
        return self.hidden_state

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

        self.multiply = base.Multiply()

        self.weighted_avg = base.WeightedAverage()

        self.layers = (
            self.lin_z,
            self.lin_r,
            self.lin_h,
            self.multiply,
            self.weighted_avg,
        )

        self.parameters = (
            *self.lin_z.parameters,
            *self.lin_r.parameters,
            *self.lin_h.parameters,
        )

    def forward(self, X):
        self._prepare_hidden_state(X)
        z = self.lin_z(self.hidden_state, X)
        r = self.lin_r(self.hidden_state, X)
        r_cs = self.multiply(r, self.hidden_state)
        h = self.lin_h(r_cs, X)
        self.hidden_state = self.weighted_avg(self.hidden_state, h, z)
        return self.hidden_state

    def backward(self, dout):
        (d_hidden_state, dh, dz) = self.weighted_avg.backward(dout)

        ((dr_cs, dXh), lin_h_params) = self.lin_h.backward(dh)

        dr, dhh = self.multiply.backward(dr_cs)

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

        self.lin_i = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Sigmoid()
        )
        self.lin_f = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Sigmoid()
        )
        self.lin_o = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Sigmoid()
        )
        self.lin_c = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, outer_activation=activation.Tanh()
        )

        self.multiply = base.Multiply()
        self.tanh = activation.Tanh()

        self.layers = (
            self.lin_i,
            self.lin_f,
            self.lin_o,
            self.lin_c,
            self.multiply,
            self.tanh,
        )

        self.parameters = (
            *self.lin_i.parameters,
            *self.lin_f.parameters,
            *self.lin_o.parameters,
            *self.lin_c.parameters,
        )

    def _prepare_cell_state(self, X):
        if self.cell_state.size == 0:
            batch_size = X.shape[0]
            self.cell_state = np.zeros((batch_size, self.dim_hidden), dtype=float)

    def forward(self, X):
        self._prepare_hidden_state(X)
        self._prepare_cell_state(X)
        i = self.lin_i(self.hidden_state, X)
        f = self.lin_f(self.hidden_state, X)
        o = self.lin_o(self.hidden_state, X)
        c = self.lin_c(self.hidden_state, X)
        self.cell_state = self.multiply(f, self.cell_state) + self.multiply(i, c)
        self.hidden_state = self.multiply(o, self.tanh(self.cell_state))
        return self.hidden_state

    def backward(self, dout, dout_cs):
        do, d_tanh_cs = self.multiply.backward(dout)
        dcs = self.tanh.backward(d_tanh_cs) + dout_cs

        di, dc = self.multiply.backward(dcs)
        df, dcs = self.multiply.backward(dcs)

        ((dhc, dXc), lin_c_params) = self.lin_c.backward(dc)
        ((dho, dXo), lin_o_params) = self.lin_o.backward(do)
        ((dhf, dXf), lin_f_params) = self.lin_f.backward(df)
        ((dhi, dXi), lin_i_params) = self.lin_i.backward(di)

        dX = dXi + dXf + dXo + dXc
        dh = dhi + dhf + dho + dhc

        return (dh, dcs, dX), (
            *lin_i_params,
            *lin_f_params,
            *lin_o_params,
            *lin_c_params,
        )

    def update(self, *args):
        if self.frozen:
            return

        (
            dWhi,
            dWii,
            dbii,
            dWhf,
            dWif,
            dbif,
            dWho,
            dWio,
            dbio,
            dWhc,
            dWic,
            dbic,
        ) = args

        self.lin_i.update(dWhi, dWii, dbii)
        self.lin_f.update(dWhf, dWif, dbif)
        self.lin_o.update(dWho, dWio, dbio)
        self.lin_c.update(dWhc, dWic, dbic)


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


class Bidirectional(base.BaseLayer):
    def __init__(self, model):
        super(Bidirectional, self).__init__(trainable=True)

        self.rnn_l_to_r = model
        self.rnn_r_to_l = model.copy()

        self.dim_hidden = int(self.rnn_l_to_r.dim_hidden)

        self.layers = (
            self.rnn_l_to_r,
            self.rnn_r_to_l,
        )

        self.parameters = (
            *self.rnn_l_to_r.parameters,
            *self.rnn_r_to_l.parameters,
        )

    def forward(self, X):
        # shape: (time, batch, dim_in)
        out_l_to_r = self.rnn_l_to_r(X)
        out_r_to_l = self.rnn_r_to_l(X[::-1, ...])
        # shape (both): (time, batch, dim_hidden)

        outputs = np.dstack((out_l_to_r, out_r_to_l))
        # shape: (time, batch, 2 * dim_hidden)

        return outputs

    def backward(self, douts):
        # if n_dim = 3: douts (time, batch, 2 * dim_hidden)
        # if n_dim = 2: douts (batch, 2 * dim_hidden)
        douts_l_to_r = self.rnn_l_to_r.backward(douts[..., : self.dim_hidden])
        douts_r_to_l = self.rnn_r_to_l.backward(douts[..., self.dim_hidden :])

        douts_X = douts_l_to_r + douts_r_to_l

        return douts_X


class DeepSequenceModel(base.BaseLayer):
    def __init__(self, model, num_layers: int):
        assert int(num_layers) > 0

        super(DeepSequenceModel, self).__init__(trainable=True)

        self.weights = compose.Sequential([model.copy() for _ in range(num_layers)])

        for model in self.weights[1:]:
            # TODO: change input dim of middle dimensions here
            pass

        self.parameters = (*self.weights.parameters,)
        self.layers = (self.weights,)

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)
