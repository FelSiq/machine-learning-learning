import numpy as np

from . import base
from . import activation
from . import compose


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

    def _prepare_cell_state(self, X):
        if self.cell_state.size == 0:
            batch_size = X.shape[0]
            self.cell_state = np.zeros((batch_size, self.dim_hidden), dtype=float)


class RNNCell(_BaseSequenceCell):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(RNNCell, self).__init__(dim_in, dim_hidden)

        self.linear = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Tanh()
        )

        self.register_layers(self.linear)

    def forward(self, X):
        self._prepare_hidden_state(X)
        self.hidden_state = self.linear(self.hidden_state, X)
        return self.hidden_state

    def backward(self, dout):
        dh, dX = self.linear.backward(dout)
        return dh, dX


class GRUCell(_BaseSequenceCell):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(GRUCell, self).__init__(dim_in, dim_hidden)

        self.lin_z = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Sigmoid()
        )
        self.lin_r = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Sigmoid()
        )
        self.lin_h = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Tanh()
        )

        self.multiply = base.Multiply()

        self.weighted_avg = base.WeightedAverage()

        self.register_layers(
            self.lin_z,
            self.lin_r,
            self.lin_h,
            self.multiply,
            self.weighted_avg,
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
        d_hidden_state, dh, dz = self.weighted_avg.backward(dout)

        dr_cs, dXh = self.lin_h.backward(dh)

        dr, dhh = self.multiply.backward(dr_cs)

        dhr, dXr = self.lin_r.backward(dr)
        dhz, dXz = self.lin_z.backward(dz)

        dh = dhz + dhr + dhh
        dX = dXz + dXr + dXh

        return dh, dX


class LSTMCell(_BaseSequenceCell):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(LSTMCell, self).__init__(dim_in, dim_hidden)

        self.lin_i = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Sigmoid()
        )
        self.lin_f = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Sigmoid()
        )
        self.lin_o = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Sigmoid()
        )
        self.lin_c = base.MultiLinear(
            [dim_hidden, dim_in], dim_hidden, activation=activation.Tanh()
        )

        self.multiply = base.Multiply()
        self.tanh = activation.Tanh()

        self.register_layers(
            self.lin_i,
            self.lin_f,
            self.lin_o,
            self.lin_c,
            self.multiply,
            self.tanh,
        )

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

        dhc, dXc = self.lin_c.backward(dc)
        dho, dXo = self.lin_o.backward(do)
        dhf, dXf = self.lin_f.backward(df)
        dhi, dXi = self.lin_i.backward(di)

        dX = dXi + dXf + dXo + dXc
        dh = dhi + dhf + dho + dhc

        return dh, dcs, dX


class Embedding(base.BaseLayer):
    def __init__(self, num_tokens: int, dim_embedding: int):
        assert int(num_tokens) > 0
        assert int(dim_embedding) > 0

        super(Embedding, self).__init__(trainable=True)

        self.embedding = base.Tensor(np.random.random((num_tokens, dim_embedding)))
        self.parameters = (self.embedding,)

    def forward(self, X):
        embedded_tokens = self.embedding.values[X, :]
        self._store_in_cache(X)
        return embedded_tokens

    def backward(self, dout):
        (orig_inds,) = self._pop_from_cache()
        dout = self.embedding.values[orig_inds, :] * dout
        d_emb = np.zeros_like(self.embedding.values)
        np.add.at(d_emb, orig_inds, dout)
        self.embedding.grads = d_emb


class Bidirectional(base.BaseLayer):
    def __init__(self, model):
        super(Bidirectional, self).__init__(trainable=True)

        self.rnn_l_to_r = model
        self.rnn_r_to_l = model.copy()

        self.dim_in = self.rnn_l_to_r.dim_in
        self.dim_hidden = self.rnn_l_to_r.dim_hidden

        self.register_layers(self.rnn_l_to_r, self.rnn_r_to_l)

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

        self.dim_in = model.dim_in
        self.dim_hidden = model.dim_hidden

        algorithm = type(model)

        self.weights = compose.Sequential(
            [
                model,
                [
                    algorithm(
                        dim_in=self.dim_hidden,
                        dim_hidden=self.dim_hidden,
                    )
                    for _ in range(num_layers - 1)
                ],
            ]
        )

        self.register_layers(self.weights)

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)
