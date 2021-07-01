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
        cs_r = self.multiply(self.hidden_state, r)
        h = self.lin_h(cs_r, X)
        self.hidden_state = self.weighted_avg(self.hidden_state, h, z)
        return self.hidden_state

    def backward(self, dout):
        d_h_avg, dh, dz = self.weighted_avg.backward(dout)

        dcsr, dXh = self.lin_h.backward(dh)

        dhh, dr = self.multiply.backward(dcsr)

        dhr, dXr = self.lin_r.backward(dr)
        dhz, dXz = self.lin_z.backward(dz)

        dh = dhz + dhr + dhh + d_h_avg
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
        self.add = base.Add()
        self.tanh = activation.Tanh()

        self.register_layers(
            self.lin_i,
            self.lin_f,
            self.lin_o,
            self.lin_c,
            self.multiply,
            self.tanh,
            self.add,
        )

    def forward(self, X):
        self._prepare_hidden_state(X)
        self._prepare_cell_state(X)

        i = self.lin_i(self.hidden_state, X)
        f = self.lin_f(self.hidden_state, X)
        o = self.lin_o(self.hidden_state, X)
        c = self.lin_c(self.hidden_state, X)

        ma = self.multiply(f, self.cell_state)
        mb = self.multiply(i, c)

        self.cell_state = self.add(ma, mb)
        self.hidden_state = self.multiply(o, self.tanh(self.cell_state))

        return self.hidden_state

    def backward(self, dout, dout_cs):
        do, d_tanh_cs = self.multiply.backward(dout)
        dcs = self.tanh.backward(d_tanh_cs) + dout_cs

        dma, dmb = self.add.backward(dcs)

        di, dc = self.multiply.backward(dmb)
        df, dcs = self.multiply.backward(dma)

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

        self.embedding = base.Tensor.from_shape(
            (num_tokens, dim_embedding), mode="normal"
        )
        self.parameters = (self.embedding,)

    def forward(self, X):
        embedded_tokens = self.embedding.values[X, :]
        self._store_in_cache(X)
        return embedded_tokens

    def backward(self, dout):
        (orig_inds,) = self._pop_from_cache()
        d_emb = np.zeros_like(self.embedding.values)
        np.add.at(d_emb, orig_inds, dout)
        self.embedding.update_grads(d_emb)


class Bidirectional(base.BaseLayer):
    def __init__(self, model):
        super(Bidirectional, self).__init__(trainable=True)

        self.rnn_l_to_r = model
        self.rnn_r_to_l = model.copy()
        self.flip = base.Flip(axis=0)

        self.dim_in = self.rnn_l_to_r.dim_in
        self.dim_hidden = self.rnn_l_to_r.dim_hidden

        self.register_layers(self.rnn_l_to_r, self.rnn_r_to_l, self.flip)

    def forward(self, X):
        # shape: (time, batch, dim_in)
        X_reversed = self.flip(X)

        out_l_to_r = self.rnn_l_to_r(X)
        out_r_to_l = self.rnn_r_to_l(X_reversed)
        # shape (both): (time, batch, dim_hidden)

        outputs = np.dstack((out_l_to_r, out_r_to_l))
        # shape: (time, batch, 2 * dim_hidden)

        return outputs

    def backward(self, douts):
        # shape: douts (time, batch, 2 * dim_hidden)
        douts_l_to_r = self.rnn_l_to_r.backward(douts[..., : self.dim_hidden])
        douts_r_to_l_reversed = self.rnn_r_to_l.backward(douts[..., self.dim_hidden :])

        douts_r_to_l = self.flip.backward(douts_r_to_l_reversed)

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


class _BasePositionalEncoding(base.BaseLayer):
    def __init__(
        self,
        max_seq_len: t.Optional[int],
        dim_in: t.Optional[int],
        trainable: bool,
        concatenate: bool,
    ):
        assert max_seq_len is None or int(max_seq_len) > 0
        assert dim_in is None or int(dim_in) > 0

        super(_BasePositionalEncoding, self).__init__(trainable=trainable)

        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else None
        self.dim_in = int(dim_in) if dim_in is not None else None
        self.concatenate = bool(concatenate)

        if not self.concatenate:
            self.add = base.Add()
            self.register_layers(self.add)

    def _get_pos_enc(self, X):
        raise NotImplementedError

    def _update_grads_pos_enc(self, dE):
        raise NotImplementedError

    def forward(self, X):
        pos_enc = self._get_pos_enc(X)

        if self.concatenate:
            out = self.concatenate((X, pos_enc), axis=-1)

        else:
            out = self.add(X, pos_enc)

        return out

    def backward(self, dout):
        if self.concatenate:
            dX, dE = np.split(dout, 2, axis=-1)

        else:
            dX, dE = self.add.backward(dout)

        self._update_grads_pos_enc(dE)

        return dX


class PositionalEncodingSinCos(_BasePositionalEncoding):
    def __init__(self, concatenate: bool = False, magic_base_period: float = 1e4):
        assert float(magic_base_period) > 0

        super(PositionalEncodingSinCos, self).__init__(
            trainable=False, concatenate=concatenate
        )

        self.magic_base_period = float(magic_base_period)
        self._cached_enc = np.empty((0, 0, 0), dtype=float)

    def _get_pos_enc(self, X):
        cache_dim_seq, _, cache_dim_enc = self._cached_enc.shape
        dim_seq, _, dim_enc = X.shape

        if cache_dim_seq <= dim_seq and cache_dim_enc <= dim_enc:
            return self._cached_enc[:dim_seq, :, :dim_enc]

        e = np.arange(1, 1 + dim_enc // 2) * -2.0 / dim_enc
        s = np.arange(1, 1 + dim_seq)

        E, S = np.meshgrid(e, s, copy=False, sparse=False)
        freqs = S * np.power(self.magic_base_period, E)

        sin = np.sin(freqs)
        cos = np.cos(freqs)

        pos_enc = np.empty((dim_seq, dim_enc), dtype=float)
        pos_enc[0:-1:2, :] = sin
        pos_enc[1::2, :] = cos

        pos_enc = np.expand_dims(pos_enc, 1)

        self._cached_enc = pos_enc

        return pos_enc

    def _update_grads_pos_enc(self, _):
        pass


class PositionalEncodingLearnable(_BasePositionalEncoding):
    def __init__(self, max_seq_len: int, dim_in: int, concatenate: bool = False):
        super(PositionalEncodingLearnable, self).__init__(
            trainable=True,
            max_seq_len=max_seq_len,
            dim_in=dim_in,
            concatenate=concatenate,
        )

        self.pos_enc = base.Tensor.from_shape((max_seq_len, 1, dim_in), mode="normal")

    def _get_pos_enc(self, X):
        return self.pos_enc.values

    def _update_grads_pos_enc(self, dE):
        self.pos_enc.update_grads(dE)
