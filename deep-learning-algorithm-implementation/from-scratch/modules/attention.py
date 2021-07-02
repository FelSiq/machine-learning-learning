import numpy as np

from . import base
from . import compose
from . import activation
from . import filters


# TODO: maybe add different forms of attention?


class Attention(base.BaseLayer):
    def __init__(
        self, mask: t.Optional[np.ndarray] = None, scale_query_key_prod: bool = True
    ):
        super(_BaseAttention, self).__init__()

        self.mask = None

        if mask is not None:
            self.mask = np.asfarray(mask)

        self.scale_query_key_prod = bool(scale_query_key_prod)

        self.softmax = activation.Softmax()
        self.multiply = base.Multiply()
        self.sum = base.Sum(axis=-1)
        self.matmul = base.Matmul(transpose_X=True)
        self.add = base.Add()

        self.register_layers(
            self.softmax, self.multiply, self.sum, self.matmul, self.add
        )

    def forward(self, Q, K, V):
        # Q shape: (???)
        # K shape: (???)
        # V shape: (???)
        att_logits = self.matmul(Q, K)

        if self.scale_query_key_prod:
            # TODO: is this right??
            att_logits /= K.shape[-1]
            self._store_in_cache(K.shape[-1])

        if self.mask is not None:
            att_logits = self.add(att_logits, self.mask)

        att_scores = self.softmax(att_logits)
        out = self.multiply(att_scores, V)
        out = self.sum(out)
        return out

    def backward(self, dout):
        dout = self.sum.backward(dout)
        d_att_scores, dV = self.multiply.backward(dout)
        d_att_logits = self.softmax.backward(d_att_scores)

        if self.mask is not None:
            d_att_logits = self.add.backward(d_att_logits)

        if self.scale_query_key_prod:
            key_dim = self._pop_from_cache()
            d_att_logits /= key_dim

        dQ, dK = self.matmul.backward(d_att_logits)

        return dQ, dK, dV


class MultiheadAttention(base.BaseLayer):
    def __init__(self, dim: int, num_heads: int):
        super(MultiheadAttention, self).__init__(trainable=True)

        self.weights = base.Linear(dim, dim, num_heads)
        # TODO: finish this.


class ConvChannelAttention2d(base.BaseLayer):
    def __init__(
        self,
        dim_in: int,
        bottleneck_ratio: float = 0.25,
        mlp_activation: t.Optional[base.BaseComponent] = None,
    ):
        assert 0.0 < float(bottleneck_ratio) <= 1.0

        super(ConvChannelAttention2d, self).__init__(trainable=True)

        bottleneck_size = max(1, int(dim_in * bottleneck_ratio))

        if mlp_activation is None:
            mlp_activation = mlp_activations.ReLU(inplace=True)

        self.mlp = compose.Sequential(
            [
                base.Flatten(),
                base.Linear(dim_in, bottleneck_size),
                mlp_activation,
                base.Linear(bottleneck_size, dim_in),
                base.Reshape(shape=(1, 1, dim_in)),
            ]
        )

        self.global_pool_max = filters.GlobalMaxPool2d()
        self.global_pool_avg = filters.GlobalAvgPool2d()
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

        mlp_max = self.mlp(X_p_max)
        mlp_avg = self.mlp(X_p_avg)

        logits = self.add(mlp_max, mlp_avg)
        weights = self.sigmoid(logits)

        out = self.multiply(X, weights)

        return out

    def backward(self, dout):
        dX_a, dW = self.multiply.backward(dout)

        dlogits = self.sigmoid.backward(dW)
        dmlp_max, dmlp_avg = self.add.backward(dlogits)

        dX_p_avg = self.mlp.backward(dmlp_avg)
        dX_p_max = self.mlp.backward(dmlp_max)

        dX_b = self.global_pool_avg.backward(dX_p_avg)
        dX_c = self.global_pool_max.backward(dX_p_max)

        dX = dX_a + dX_b + dX_c

        return dX


class ConvSpatialAttention2d(base.BaseLayer):
    def __init__(
        self,
        activation_conv: t.Optional[base.BaseComponent] = None,
        norm_layer_after_conv: t.Optional[base.BaseComponent] = None,
    ):
        super(ConvSpatialAttention2d, self).__init__(trainable=True)
        self.chan_pool_max = filters.ChannelMaxPool2d()
        self.chan_pool_avg = filters.ChannelAvgPool2d()
        self.dconcat = filters.Concatenate(axis=2)
        self.conv2d = filters.Conv2d(
            channels_in=2,
            channels_out=1,
            padding_mode="same",
            activation=activation_conv,
        )
        self.sigmoid = activation.Sigmoid()

        self.register_layers(
            self.chan_pool_max,
            self.chan_pool_avg,
            self.dconcat,
            self.sigmoid,
            self.conv2d,
        )

        if norm_layer_after_conv is not None:
            self.norm_layer_after_conv = norm_layer_after_conv
            self.register_layers(self.norm_layer_after_conv)

    def forward(self, X):
        C_p_max = self.chan_pool_max(X)
        C_p_avg = self.chan_pool_avg(X)
        C = self.dconcat(C_p_max, C_p_avg)
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
        dC_p_max, dC_p_avg = self.dconcat.backward(dC)
        dX_a = self.chan_pool_avg.backward(dC_p_avg)
        dX_b = self.chan_pool_max.backward(dX_p_max)

        dX = dX_a + dX_b

        return dX


class ConvBlockAttention2d(base.BaseLayer):
    def __init__(
        self,
        dim_in: int,
        channel_bottleneck_ratio: float = 0.25,
        channel_mlp_activation: t.Optional[base.BaseComponent] = None,
        spatial_activation_conv: t.Optional[base.BaseComponent] = None,
        spatial_norm_layer_after_conv: t.Optional[base.BaseComponent] = None,
    ):
        super(ConvBlockAttention2d, self).__init__(trainable=True)

        self.weights = compose.Sequential(
            [
                compose.SkipConnection(
                    layer_main=ConvChannelAttention2d(
                        dim_in=dim_in,
                        bottleneck_ratio=channel_bottleneck_ratio,
                        mlp_activation=channel_mlp_activation,
                    ),
                    layer_combine=base.Multiply(),
                ),
                compose.SkipConnection(
                    layer_main=ConvSpatialAttention2d(
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
        dim_in: int,
        bottleneck_ratio: float = 0.25,
        mlp_activation: t.Optional[base.BaseComponent] = None,
    ):
        assert 0.0 < float(bottleneck_ratio) <= 1.0
        super(SqueezeExcite, self).__init__(trainable=True)

        dim_in = int(dim_in)
        bottleneck_size = max(1, int(dim_in * bottleneck_ratio))

        if mlp_activation is None:
            mlp_activation = activations.ReLU(inplace=True)

        self.weights = compose.SkipConnection(
            layer_main=compose.Sequential(
                [
                    filters.GlobalAvgPool2d(squeeze=True),
                    base.Linear(dim_in, bottleneck_size),
                    mlp_activation,
                    base.Linear(bottleneck_size, dim_in),
                    base.Reshape(shape=(1, 1, dim_in)),
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
