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

    @staticmethod
    def crop_center(X, out_shape, discard_left_if_truncate: bool = False):
        out = X
        dyn_ignored_axes = [slice(None)] * len(out_shape)

        for i, s_out in enumerate(out_shape):
            s_in = X.shape[i]
            diff = s_in - s_out

            assert diff >= 0

            l_excess = diff // 2
            r_excess = (diff + 1) // 2

            if discard_left_if_truncate:
                l_excess, r_excess = r_excess, l_excess

            dyn_ignored_axes[i] = slice(l_excess, s_in - r_excess)
            out = out[tuple(dyn_ignored_axes)]
            dyn_ignored_axes[i] = slice(None)

        return out


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

        assert _utils.all_nonzero(stride)

        self.num_spatial_dims = int(num_spatial_dims)

        self.stride = _utils.replicate(stride, num_spatial_dims)
        self.stride = tuple(
            s if s > 0 else k for s, k in zip(self.stride, self.kernel_size)
        )

        assert len(self.stride) == num_spatial_dims

    def calc_out_spatial_dim(self, X: int, dim: int):
        stride = self.stride[dim]
        kernel_size = self.kernel_size[dim]
        input_dim = X.shape[dim + 1]
        return 1 + (input_dim - kernel_size) // stride


class _BaseFixedFilter(_BaseFilter):
    pass


class _BaseConv(_BaseMovingFilter):
    def __init__(
        self,
        num_spatial_dims: int,
        channels_in: int,
        channels_out: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
        padding_type: str = "valid",
        padding_mode: str = "constant",
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
    ):
        assert int(channels_in) > 0
        assert int(channels_out) > 0
        assert str(padding_type) in {"valid", "same"}

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
        padding = self.pad_widths[dim + 1][0] if self.use_padding else 0
        return 1 + (input_dim + 2 * padding - kernel_size) // stride


class LearnableFilter2d(_BaseFixedFilter):
    def __init__(
        self,
        dim_in: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        include_bias: bool = True,
        reduce_layer: t.Optional[base.BaseComponent] = base.Sum,
    ):
        assert int(dim_in) > 0

        super(LearnableFilter2d, self).__init__(
            num_spatial_dims=2,
            trainable=True,
            kernel_size=kernel_size,
        )

        self.weights = base.Tensor.from_shape(
            (1, *self.kernel_size, int(dim_in)),
            mode="uniform",
            std=("xavier", float(dim_in * np.prod(self.kernel_size))),
        )
        self.bias = base.Tensor()

        if include_bias:
            self.bias = base.Tensor.from_shape((1,), mode="zeros")
            self.parameters = (self.weights, self.bias)

        else:
            self.parameters = (self.weights,)

        if reduce_layer is None:
            reduce_layer = base.Identity

        self.reduce_layer = reduce_layer(
            axis=tuple(range(1, len(self.weights.shape))),
            enforce_batch_dim=False,
            keepdims=False,
        )

        self.mult = base.Multiply()
        self.add = base.Add()
        self.register_layers(self.reduce_layer, self.mult, self.add)

    def forward(self, X):
        out = self.mult(X, self.weights.values)
        out = self.reduce_layer(out)

        if self.bias.size:
            out = self.add(out, self.bias.values)

        self._store_in_cache(X)

        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        if self.bias.size:
            dout, db = self.add.backward(dout)
            self.bias.update_grads(db)

        dout = self.reduce_layer.backward(dout)
        dX, dW = self.mult.backward(dout)

        self.weights.update_grads(dW)

        return dX


class MaxFilter2d(_BaseFixedFilter):
    def __init__(
        self,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
    ):
        super(MaxFilter2d, self).__init__(
            kernel_size=kernel_size,
            num_spatial_dims=2,
            trainable=False,
        )

    def forward(self, X):
        input_shape = X.shape

        _, h_dim, w_dim, d_dim = input_shape

        X = X.reshape(-1, h_dim * w_dim, d_dim)

        max_indices = np.argmax(X, axis=1)
        out = np.take_along_axis(X, np.expand_dims(max_indices, 1), axis=1)
        out = np.squeeze(out)

        if out.ndim == 1:
            out = np.expand_dims(out, -1)

        self._store_in_cache(max_indices, input_shape)

        return out

    def backward(self, dout):
        (max_indices, input_shape) = self._pop_from_cache()

        _, h_dim, w_dim, d_dim = input_shape

        max_indices = max_indices.reshape(-1, 1, d_dim)

        dout_b = max_indices == np.arange(h_dim * w_dim).reshape(1, -1, 1)
        dout = np.expand_dims(dout, (1, 2))
        dout_b = dout_b.astype(float, copy=False).reshape(input_shape) * dout

        return dout_b


class Conv2d(_BaseConv):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
        padding_type: str = "valid",
        padding_mode: str = "constant",
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

        h_out_dim = self.calc_out_spatial_dim(X, 0)
        w_out_dim = self.calc_out_spatial_dim(X, 1)

        if self.use_padding:
            X = np.pad(X, pad_width=self.pad_widths, mode=self.padding_mode)

        input_shape_padded = X.shape

        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size
        batch_size, h_dim, w_dim, d_dim = X.shape

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

        self._store_in_cache(input_shape, input_shape_padded)

        return out

    def backward(self, dout):
        (input_shape, input_shape_padded) = self._pop_from_cache()

        if self.activation is not None:
            dout = self.activation.backward(dout)

        dout_b = np.zeros(input_shape_padded, dtype=float)

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
                    filter_grad = filter_.backward(dout[:, r, c, d])
                    dout_b[:, h_start:h_end, w_start:w_end, :] += filter_grad

        if self.use_padding:
            dout_b = self.crop_center(dout_b, input_shape)

        return dout_b


class ConvTranspose2d(_BaseConv):
    pass


class _Pool2d(_BaseMovingFilter):
    def __init__(
        self,
        trainable: bool,
        filter_: _BaseFixedFilter,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = -1,
    ):
        super(_Pool2d, self).__init__(
            num_spatial_dims=2,
            trainable=trainable,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.filter = filter_
        self.register_layers(self.filter)

    def forward(self, X):
        h_out_dim = self.calc_out_spatial_dim(X, 0)
        w_out_dim = self.calc_out_spatial_dim(X, 1)

        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size
        batch_size, h_dim, w_dim, d_dim = X.shape

        out = np.empty((batch_size, h_out_dim, w_out_dim, d_dim), dtype=float)

        for r, h_start in enumerate(range(0, h_dim - h_kernel + 1, h_stride)):
            h_end = h_start + h_kernel
            for c, w_start in enumerate(range(0, w_dim - w_kernel + 1, w_stride)):
                w_end = w_start + w_kernel

                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                out[:, r, c, :] = self.filter(X_slice)

        self._store_in_cache(X.shape)

        return out

    def backward(self, dout):
        (input_shape,) = self._pop_from_cache()

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

                grad = self.filter.backward(dout[:, r, c, :])
                dout_b[:, h_start:h_end, w_start:w_end, :] += grad

        return dout_b


class MaxPool2d(_Pool2d):
    def __init__(
        self,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = -1,
    ):
        super(MaxPool2d, self).__init__(
            trainable=False,
            filter_=MaxFilter2d(kernel_size=kernel_size),
            kernel_size=kernel_size,
            stride=stride,
        )


class AvgPool2d(_Pool2d):
    def __init__(
        self,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = -1,
    ):
        super(AvgPool2d, self).__init__(
            trainable=False,
            filter_=base.Average(axis=(1, 2), keepdims=False),
            kernel_size=kernel_size,
            stride=stride,
        )
