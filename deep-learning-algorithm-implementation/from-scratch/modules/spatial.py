import typing as t

import numpy as np

from . import base
from . import _utils


class _BaseFilter(base.BaseLayer):
    def __init__(
        self,
        num_spatial_dims: int,
        trainable: bool,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
    ):
        super(_BaseFilter, self).__init__(trainable=trainable)

        assert int(num_spatial_dims) > 0
        assert _utils.all_positive(kernel_size)

        self.kernel_size = _utils.replicate(kernel_size, num_spatial_dims)

        assert len(self.kernel_size) == num_spatial_dims


class _BaseMovingFilter(_BaseFilter):
    def __init__(
        self,
        num_spatial_dims: int,
        trainable: bool,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
    ):
        super(_BaseMovingFilter, self).__init__(
            num_spatial_dims=num_spatial_dims,
            trainable=trainable,
            kernel_size=kernel_size,
        )

        assert _utils.all_positive(stride)

        self.num_spatial_dims = int(num_spatial_dims)
        self.stride = _utils.replicate(stride, num_spatial_dims)

        assert len(self.stride) == num_spatial_dims


class _BaseConv(_BaseMovingFilter):
    def __init__(
        self,
        num_spatial_dims: int,
        channels_in: int,
        channels_out: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
        padding_type: str = "valid",
        padding_mode: str = "zeros",
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
    ):
        assert int(channels_in) > 0
        assert int(channels_out) > 0
        assert str(padding_type) in {"valid", "same"}
        assert str(padding_mode) in {"zeros", "reflect", "replicate", "circular"}

        super(_BaseConv, self).__init__(
            num_spatial_dims=num_spatial_dims,
            kernel_size=kernel_size,
            stride=stride,
            trainable=True,
        )

        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.use_padding = str(padding_type) == "same"
        self.padding_mode = str(padding_mode)

        self.pad_widths = [(k // 2, k // 2) for k in self.kernel_size]
        self.pad_widths.insert(0, (0, 0))
        self.pad_widths.append((0, 0))

        self.activation = activation

        if self.activation is not None:
            self.register_layers(self.activation)

    def calc_out_spatial_dim(self, X: int, dim: int):
        stride = self.stride[dim]
        kernel_size = self.kernel_size[dim]
        input_dim = X.shape[dim + 1]
        padding = self.pad_widths[dim][0] if self.use_padding else 0
        return 1 + (input_dim + 2 * padding - kernel_size) // stride


class _BaseDropout(base.BaseLayer):
    def __init__(self, prob: float):
        assert 1.0 >= float(prob) >= 0.0
        super(_BaseDropout, self).__init__()
        self.prob = float(prob)


class LearnableFilter2d(_BaseFilter):
    def __init__(
        self,
        dim_in: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        include_bias: bool = True,
    ):
        assert int(dim_in) > 0

        super(LearnableFilter2d, self).__init__(
            num_spatial_dims=2,
            trainable=True,
            kernel_size=kernel_size,
        )

        self.weights = base.Tensor(np.random.randn(1, *self.kernel_size, int(dim_in)))
        self.bias = base.Tensor()

        if include_bias:
            self.bias = base.Tensor(np.zeros(1, dtype=float))

        self.parameters = (
            self.weights,
            self.bias,
        )

        self._sum_axes = tuple(range(1, len(self.weights.shape)))

    def forward(self, X):
        out = np.sum(X * self.weights.values, axis=self._sum_axes) + self.bias.values
        self._store_in_cache(X)
        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        dout = np.expand_dims(dout, self._sum_axes)

        dX = np.sum(self.weights.values * dout, axis=0)
        dW = np.sum(X * dout, axis=0)

        self.weights.grads = dW

        if self.bias.size:
            db = np.expand_dims(np.sum(dout), 0)
            self.bias.grads = db

        return dX


class Conv2d(_BaseConv):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
        padding_type: str = "valid",
        padding_mode: str = "zeros",
        include_bias: bool = True,
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super(Conv2d, self).__init__(
            num_spatial_dims=2,
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding_type=padding_type,
            padding_mode=padding_mode,
            activation=activation,
        )

        self.filters = [
            LearnableFilter2d(
                dim_in=self.channels_in,
                kernel_size=self.kernel_size,
                include_bias=include_bias,
            )
            for _ in range(self.channels_out)
        ]

        self.register_layers(*self.filters)

    def forward(self, X):
        input_shape = X.shape

        if self.use_padding:
            X = np.pad(X, pad_width=self.pad_widths, mode=self.padding_mode)

        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size
        batch_size, h_dim, w_dim, d_dim = X.shape

        h_out_dim = self.calc_out_spatial_dim(X, 0)
        w_out_dim = self.calc_out_spatial_dim(X, 1)

        out = np.empty(
            (batch_size, h_out_dim, w_out_dim, self.channels_out), dtype=float
        )

        for r, h_start in enumerate(range(0, h_dim - h_kernel + 1, h_stride)):
            h_end = h_start + h_kernel
            for c, w_start in enumerate(range(0, w_dim - w_kernel + 1, w_stride)):
                w_end = w_start + w_kernel

                X_slice = X[:, h_start:h_end, w_start:w_end, :]

                for f_ind, filter_ in enumerate(self.filters):
                    out[:, r, c, f_ind] = filter_(X_slice)

        if self.activation is not None:
            out = self.activation(out)

        self._store_in_cache(input_shape)

        return out

    def backward(self, dout):
        (input_shape,) = self._pop_from_cache()

        if self.activation is not None:
            dout = self.activation.backward(dout)

        dout_b = np.zeros(input_shape, dtype=float)

        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size
        _, h_dout, w_dout, d_dout = dout.shape

        for r in reversed(range(h_dout)):
            h_start = r * h_stride
            h_end = h_start + h_kernel
            for c in reversed(range(w_dout)):
                w_start = c * w_stride
                w_end = w_start + w_kernel
                for d in reversed(range(d_dout)):
                    filter_ = self.filters[d]
                    dout_b[:, h_start:h_end, w_start:w_end, :] += filter_.backward(
                        dout[:, r, c, d]
                    )

        return dout_b


class Dropout2d(_BaseDropout):
    pass


class MaxPool2d(_BaseMovingFilter):
    pass


class AvgPool2d(_BaseMovingFilter):
    pass
