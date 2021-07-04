import typing as t

import numpy as np

from . import base
from . import compose
from . import activation
from . import filter as filter_


class AttentionQKV(base.BaseLayer):
    def __init__(self, scale_query_key_prod: bool = True):
        super(AttentionQKV, self).__init__()

        self.scale_query_key_prod = bool(scale_query_key_prod)

        self.softmax = activation.Softmax(axis=2)
        self.matmul_Q_K = base.Matmul(transpose_X=True)
        self.matmul_scores_V = base.Matmul(transpose_X=False)
        self.add = base.Add()
        self.permute_batch_first = base.PermuteAxes((1, 2, 0))
        self.permute_seq_first = base.PermuteAxes((2, 0, 1))

        self.register_layers(
            self.softmax,
            self.matmul_Q_K,
            self.matmul_scores_V,
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
        # Q shape: (L, N, E)
        # K shape: (S, N, E)
        # V shape: (S, N, E)

        Q_batch_first = self.permute_batch_first(Q)
        K_batch_first = self.permute_batch_first(K)
        V_batch_first = self.permute_batch_first(V)

        att_logits = self.matmul_Q_K(Q_batch_first, K_batch_first)
        # att_Logits shape: (N, E, L) . (N, E, S) = (N, L, S)

        if self.scale_query_key_prod:
            query_dim = Q.shape[-1]
            att_logits /= query_dim
            self._store_in_cache(query_dim)

        if mask is not None:
            # mask shape: (L, S)
            att_logits = self.add(att_logits, mask)

        att_scores = self.softmax(att_logits)
        out_batch_first = self.matmul_scores_V(V_batch_first, att_scores)
        out = self.permute_seq_first(out_batch_first)
        # out shape: (S, N, E)

        return out

    def backward(self, dout):
        dout_batch_first = self.permute_seq_first.backward(dout)
        dV_batch_first, datt_scores = self.matmul_scores_V.backward(dout_batch_first)
        datt_logits = self.softmax.backward(datt_scores)

        if self.add.has_stored_grads:
            datt_logits = self.add.backward(datt_logits)

        if self.scale_query_key_prod:
            (query_dim,) = self._pop_from_cache()
            datt_logits /= query_dim

        dQ_batch_first, dK_batch_first = self.matmul_Q_K.backward(datt_logits)

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

    def forward(self, X, Y=None, mask=None):
        self._store_in_cache(Y is None)

        if Y is None:
            Y = X

        # X shape: (seq, batch, emb)
        # Y shape: (seq, batch, emb)

        Q = self.weights_query(Y)
        K = self.weights_key(X)
        V = self.weights_value(X)

        # Q shape: (seq, batch, n_heads * emb)
        # K shape: (seq, batch, n_heads * emb)
        # V shape: (seq, batch, n_heads * emb)

        out = self.attentionQKV(Q, K, V, mask=mask)
        out = self.weights_out(out)

        return out

    def backward(self, dout):
        (is_self_attention,) = self._pop_from_cache()

        dout = self.weights_out.backward(dout)
        dQ, dK, dV = self.attentionQKV.backward(dout)
        dX_a = self.weights_value.backward(dV)
        dX_b = self.weights_key.backward(dK)
        dY = self.weights_query.backward(dQ)

        dX = dX_a + dX_b

        if is_self_attention:
            dX += dY
            return dX

        return dX, dY


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
        self.multiply = base.Multiply()

        self.register_layers(
            self.mlp,
            self.global_pool_max,
            self.global_pool_avg,
            self.add,
            self.sigmoid,
            self.multiply,
        )

    def forward(self, X):
        C_p_max = self.global_pool_max(X)
        C_p_avg = self.global_pool_avg(X)

        mlp_max = self.mlp(C_p_max)
        mlp_avg = self.mlp(C_p_avg)

        logits = self.add(mlp_max, mlp_avg)
        weights = self.sigmoid(logits)

        out = self.multiply(X, weights)

        return out

    def backward(self, dout):
        dX_a, dW = self.multiply.backward(dout)

        dlogits = self.sigmoid.backward(dW)
        dmlp_max, dmlp_avg = self.add.backward(dlogits)

        dC_p_avg = self.mlp.backward(dmlp_avg)
        dC_p_max = self.mlp.backward(dmlp_max)

        dX_b = self.global_pool_avg.backward(dC_p_avg)
        dX_c = self.global_pool_max.backward(dC_p_max)

        dX = dX_a + dX_b + dX_c

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
