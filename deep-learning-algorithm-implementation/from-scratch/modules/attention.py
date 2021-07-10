import typing as t

import numpy as np

from . import base
from . import compose
from . import activation
from . import filter as filter_


class AttentionQKV(base.BaseLayer):
    def __init__(self, scale_query_key_prod: bool = True, batch_first: bool = False):
        super(AttentionQKV, self).__init__()

        self.scale_query_key_prod = bool(scale_query_key_prod)

        self.softmax = activation.Softmax(axis=2)
        self.matmul_Q_K = base.Matmul(transpose_X=True)
        self.matmul_V_scores = base.Matmul(transpose_Y=True)
        self.add = base.Add()
        self.permute_batch_first = base.PermuteAxes((1, 2, 0))
        self.permute_seq_first = base.PermuteAxes((2, 0, 1))
        self.batch_first = bool(batch_first)

        self.register_layers(
            self.softmax,
            self.matmul_Q_K,
            self.matmul_V_scores,
            self.add,
            self.permute_batch_first,
            self.permute_seq_first,
        )

    def __call__(self, Q, K, V, mask: t.Optional[np.ndarray] = None):
        return self.forward(Q, K, V, mask=mask)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: t.Optional[np.ndarray] = None,
    ):
        if self.batch_first:
            Q_batch_first = Q
            K_batch_first = K
            V_batch_first = V

        else:
            # Q shape: (L, N, E)
            # K shape: (S, N, E)
            # V shape: (S, N, E)
            Q_batch_first = self.permute_batch_first(Q)
            K_batch_first = self.permute_batch_first(K)
            V_batch_first = self.permute_batch_first(V)

        # Q_batch_first shape: (N, E, L)
        # K_batch_first shape: (N, E, S)
        # V_batch_first shape: (N, E, S)

        att_logits = self.matmul_Q_K(Q_batch_first, K_batch_first)
        # att_Logits shape: (N, E, L) . (N, E, S) = (N, L, S)

        if self.scale_query_key_prod:
            sqrt_key_dim = np.sqrt(K.shape[-1])
            att_logits /= sqrt_key_dim
            self._store_in_cache(sqrt_key_dim)

        if mask is not None:
            # mask shape: (L, S)
            att_logits = self.add(att_logits, mask)

        att_scores = self.softmax(att_logits)
        out_batch_first = self.matmul_V_scores(V_batch_first, att_scores)

        if self.batch_first:
            out = out_batch_first

        else:
            # out_batch_first shape: (N, E, L)
            out = self.permute_seq_first(out_batch_first)

        # out shape: (L, N, E)
        return out

    def backward(self, dout):
        if self.batch_first:
            dout_batch_first = dout

        else:
            dout_batch_first = self.permute_seq_first.backward(dout)

        dV_batch_first, datt_scores = self.matmul_V_scores.backward(dout_batch_first)
        datt_logits = self.softmax.backward(datt_scores)

        if self.add.has_stored_grads:
            datt_logits, _ = self.add.backward(datt_logits)

        if self.scale_query_key_prod:
            (sqrt_key_dim,) = self._pop_from_cache()
            datt_logits /= sqrt_key_dim

        dQ_batch_first, dK_batch_first = self.matmul_Q_K.backward(datt_logits)

        if self.batch_first:
            dQ = dQ_batch_first
            dK = dK_batch_first
            dV = dV_batch_first

        else:
            dQ = self.permute_batch_first.backward(dQ_batch_first)
            dK = self.permute_batch_first.backward(dK_batch_first)
            dV = self.permute_batch_first.backward(dV_batch_first)

        return dQ, dK, dV


class MultiheadAttentionQKV(base.BaseLayer):
    def __init__(self, dim_in: int, n_heads: int, scale_query_key_prod: bool = True):
        assert int(n_heads) > 0

        super(MultiheadAttentionQKV, self).__init__(trainable=True)

        dtnh = dim_in * n_heads

        self.n_heads = int(n_heads)

        self.weights_query = base.Linear(dim_in, dtnh)
        self.weights_key = base.Linear(dim_in, dtnh)
        self.weights_value = base.Linear(dim_in, dtnh)
        self.weights_out = base.Linear(dtnh, dim_in)
        self.attentionQKV = AttentionQKV(scale_query_key_prod=scale_query_key_prod)

        self.register_layers(
            self.weights_query,
            self.weights_key,
            self.weights_value,
            self.weights_out,
            self.attentionQKV,
        )

    def __call__(self, X, Y=None, mask=None):
        return self.forward(X, Y, mask)

    def forward(self, X, Y=None, mask=None):
        self._store_in_cache(Y is None)

        if Y is None:
            Y = X

        # X shape: (L, batch, emb)
        # Y shape: (S, batch, emb)

        Q = self.weights_query(X)
        K = self.weights_key(Y)
        V = self.weights_value(Y)

        # Q shape: (S, batch, n_heads * emb)
        # K shape: (L, batch, n_heads * emb)
        # V shape: (L, batch, n_heads * emb)

        out = self.attentionQKV(Q, K, V, mask=mask)
        out = self.weights_out(out)

        return out

    def backward(self, dout):
        (is_self_attention,) = self._pop_from_cache()

        dout = self.weights_out.backward(dout)
        dQ, dK, dV = self.attentionQKV.backward(dout)

        dY_a = self.weights_value.backward(dV)
        dY_b = self.weights_key.backward(dK)
        dY = dY_a + dY_b

        dX = self.weights_query.backward(dQ)

        if is_self_attention:
            dY += dX
            return dY

        return dX, dY


class MultiheadMaskedSelfAttentionQKV(base.BaseLayer):
    def __init__(
        self,
        dim_in: int,
        n_heads: int,
        scale_query_key_prod: bool = True,
        mask_type: str = "upper",
    ):
        assert str(mask_type) in {"lower", "upper"}
        super(MultiheadMaskedSelfAttentionQKV, self).__init__(trainable=True)
        self.mha_qkv = MultiheadAttentionQKV(
            dim_in=dim_in, n_heads=n_heads, scale_query_key_prod=scale_query_key_prod
        )
        self.register_layers(self.mha_qkv)
        self.mask_lower = str(mask_type) == "lower"

    def forward(self, X):
        # X shape: (T, N, E)
        dim_seq, _, _ = X.shape
        mask = np.zeros((dim_seq, dim_seq), dtype=float)
        # mask shape: (T, T)
        mask_fun, offset = (
            (np.tril_indices_from, -1) if self.mask_lower else (np.triu_indices_from, 1)
        )
        mask[mask_fun(mask, k=offset)] = -np.inf
        out = self.mha_qkv(X, mask=mask)
        return out

    def backward(self, dout):
        return self.mha_qkv.backward(dout)


class ConvChannelAttention2d(base.BaseLayer):
    def __init__(
        self,
        channels_in: int,
        bottleneck_ratio: float = 0.20,
        mlp_activation: t.Optional[base.BaseComponent] = None,
    ):
        assert 0.0 < float(bottleneck_ratio) <= 1.0

        super(ConvChannelAttention2d, self).__init__(trainable=True)

        bottleneck_size = max(1, int(channels_in * bottleneck_ratio))

        if mlp_activation is None:
            mlp_activation = activation.ReLU(inplace=True)

        self.mlp = compose.Sequential(
            [
                base.Flatten(),
                base.Linear(channels_in, bottleneck_size),
                mlp_activation,
                base.Linear(bottleneck_size, channels_in),
                base.Reshape(out_shape=(1, 1, channels_in)),
            ]
        )

        self.global_pool_max = filter_.GlobalMaxPool2d()
        self.global_pool_avg = filter_.GlobalAvgPool2d()
        self.add = base.Add()
        self.sigmoid = activation.Sigmoid()

        self.register_layers(
            self.mlp,
            self.global_pool_max,
            self.global_pool_avg,
            self.add,
            self.sigmoid,
        )

    def forward(self, X):
        C_p_max = self.global_pool_max(X)
        C_p_avg = self.global_pool_avg(X)

        mlp_max = self.mlp(C_p_max)
        mlp_avg = self.mlp(C_p_avg)

        logits = self.add(mlp_max, mlp_avg)
        out = self.sigmoid(logits)

        return out

    def backward(self, dout):
        dlogits = self.sigmoid.backward(dout)
        dmlp_max, dmlp_avg = self.add.backward(dlogits)

        dC_p_avg = self.mlp.backward(dmlp_avg)
        dC_p_max = self.mlp.backward(dmlp_max)

        dX_a = self.global_pool_avg.backward(dC_p_avg)
        dX_b = self.global_pool_max.backward(dC_p_max)

        dX = dX_a + dX_b

        return dX


class ConvSpatialAttention2d(base.BaseLayer):
    def __init__(
        self,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        activation_conv: t.Optional[base.BaseComponent] = None,
        norm_layer_after_conv: t.Optional[base.BaseComponent] = None,
    ):
        super(ConvSpatialAttention2d, self).__init__(trainable=True)

        self.chan_pool_max = filter_.ChannelMaxPool2d()
        self.chan_pool_avg = filter_.ChannelAvgPool2d()
        self.chan_concat = base.Concatenate(axis=3)
        self.conv2d = filter_.Conv2d(
            channels_in=2,
            channels_out=1,
            kernel_size=kernel_size,
            padding_type="same",
            activation=activation_conv,
        )
        self.sigmoid = activation.Sigmoid()

        self.register_layers(
            self.chan_pool_max,
            self.chan_pool_avg,
            self.chan_concat,
            self.sigmoid,
            self.conv2d,
        )

        self.norm_layer_after_conv = None

        if norm_layer_after_conv is not None:
            self.norm_layer_after_conv = norm_layer_after_conv
            self.register_layers(self.norm_layer_after_conv)

    def forward(self, X):
        C_p_max = self.chan_pool_max(X)
        C_p_avg = self.chan_pool_avg(X)

        C = self.chan_concat(C_p_max, C_p_avg)
        logits = self.conv2d(C)

        if self.norm_layer_after_conv is not None:
            logits = self.norm_layer_after_conv(logits)

        out = self.sigmoid(logits)

        return out

    def backward(self, dout):
        dlogits = self.sigmoid.backward(dout)

        if self.norm_layer_after_conv is not None:
            dlogits = self.norm_layer_after_conv.backward(dlogits)

        dC = self.conv2d.backward(dlogits)
        dC_p_max, dC_p_avg = self.chan_concat.backward(dC)
        dX_a = self.chan_pool_avg.backward(dC_p_avg)
        dX_b = self.chan_pool_max.backward(dC_p_max)

        dX = dX_a + dX_b

        return dX


class ConvBlockAttention2d(base.BaseLayer):
    def __init__(
        self,
        channels_in: int,
        kernel_size: int,
        channel_bottleneck_ratio: float = 0.20,
        channel_mlp_activation: t.Optional[base.BaseComponent] = None,
        spatial_activation_conv: t.Optional[base.BaseComponent] = None,
        spatial_norm_layer_after_conv: t.Optional[base.BaseComponent] = None,
    ):
        super(ConvBlockAttention2d, self).__init__(trainable=True)

        self.weights = compose.Sequential(
            [
                compose.SkipConnection(
                    layer_main=ConvChannelAttention2d(
                        channels_in=channels_in,
                        bottleneck_ratio=channel_bottleneck_ratio,
                        mlp_activation=channel_mlp_activation,
                    ),
                    layer_combine=base.Multiply(),
                ),
                compose.SkipConnection(
                    layer_main=ConvSpatialAttention2d(
                        kernel_size=kernel_size,
                        activation_conv=spatial_activation_conv,
                        norm_layer_after_conv=spatial_norm_layer_after_conv,
                    ),
                    layer_combine=base.Multiply(),
                ),
            ]
        )

        self.register_layers(self.weights)

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)


class SqueezeExcite(base.BaseLayer):
    def __init__(
        self,
        channels_in: int,
        bottleneck_ratio: float = 0.20,
        mlp_activation: t.Optional[base.BaseComponent] = None,
    ):
        assert 0.0 < float(bottleneck_ratio) <= 1.0
        super(SqueezeExcite, self).__init__(trainable=True)

        channels_in = int(channels_in)
        bottleneck_size = max(1, int(channels_in * bottleneck_ratio))

        if mlp_activation is None:
            mlp_activation = activation.ReLU(inplace=True)

        self.weights = compose.SkipConnection(
            layer_main=compose.Sequential(
                [
                    filter_.GlobalAvgPool2d(keepdims=False),
                    base.Linear(channels_in, bottleneck_size),
                    mlp_activation,
                    base.Linear(bottleneck_size, channels_in),
                    base.Reshape(out_shape=(1, 1, channels_in)),
                    activation.Sigmoid(),
                ]
            ),
            layer_combine=base.Multiply(),
        )

        self.register_layers(self.weights)

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)


class ConvAttentionQKV2d(base.BaseLayer):
    def __init__(
        self, channels_in: int, n_heads: int, scale_query_key_prod: bool = True
    ):
        super(ConvAttentionQKV2d, self).__init__(trainable=True)

        self.conv_in = filter_.Conv2d(
            channels_in=channels_in, channels_out=3 * n_heads, kernel_size=1
        )
        self.conv_out = filter_.Conv2d(
            channels_in=n_heads, channels_out=channels_in, kernel_size=1
        )
        self.chan_split = base.Split(3, axis=1)
        self.attention_qkv = AttentionQKV(
            scale_query_key_prod=scale_query_key_prod, batch_first=True
        )
        self.reshape_collapse = base.CollapseAdjacentAxes(axis_first=1, axis_last=2)
        self.reshape_expand = base.Reshape()

        self.chan_swap_axes = base.PermuteAxes((0, 2, 1))

        self.channels_in = int(channels_in)

        self.register_layers(
            self.conv_in,
            self.conv_out,
            self.chan_split,
            self.attention_qkv,
            self.reshape_collapse,
            self.reshape_expand,
            self.chan_swap_axes,
        )

    def forward(self, X):
        # aux shape: (batch, height, width, channels_in)
        aux = self.conv_in(X)
        # aux shape: (batch, height, width, 3 * n_heads)
        aux_collapsed = self.reshape_collapse(aux)
        # aux_collapsed shape: (batch, height * width, 3 * n_heads)
        aux_chan_first = self.chan_swap_axes(aux_collapsed)
        # aux_collapsed shape: (batch, 3 * n_heads, height * width)
        Q, K, V = self.chan_split(aux_chan_first)
        # Q shape = K shape = V shape: (batch, n_heads, height * width)
        att_heads_out_chan_first = self.attention_qkv(Q, K, V)
        # att_heads_out_chan_first shape: (batch, n_heads, height * width)
        att_heads_out_collapsed = self.chan_swap_axes(att_heads_out_chan_first)
        # att_heads_out_collapsed shape: (batch, height * width, n_heads)
        att_heads_out = self.reshape_expand(
            att_heads_out_collapsed, (*X.shape[:3], Q.shape[1])
        )
        # att_heads_out shape: (batch_first, height, width, n_heads)
        out = self.conv_out(att_heads_out)
        # out shape: (batch_first, height, width, channels_in)
        return out

    def backward(self, dout):
        datt_heads_out = self.conv_out.backward(dout)
        datt_heads_out_collapsed = self.reshape_expand.backward(datt_heads_out)
        datt_heads_out_chan_first = self.chan_swap_axes.backward(
            datt_heads_out_collapsed
        )
        dQdKdV = self.attention_qkv.backward(datt_heads_out_chan_first)
        daux_chan_first = self.chan_split.backward(dQdKdV)
        daux_collapsed = self.chan_swap_axes.backward(daux_chan_first)
        daux = self.reshape_collapse.backward(daux_collapsed)
        dX = self.conv_in.backward(daux)
        return dX
