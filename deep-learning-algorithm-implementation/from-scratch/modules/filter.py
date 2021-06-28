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

    def calc_out_spatial_dim(self, X: int, dim: int, transpose: bool = False):
        stride = self.stride[dim]
        kernel_size = self.kernel_size[dim]
        input_dim = X.shape[dim + 1]
        padding = self.pad_widths[dim + 1][0] if self.use_padding else 0

        if transpose:
            # Note: ignoring the padding in the formula since we will remove
            # the padding later on, and it is useful to keep the padding for now.
            # return (input_dim - 1) * stride - 2 * padding + kernel_size
            return (input_dim - 1) * stride + kernel_size

        return 1 + (input_dim + 2 * padding - kernel_size) // stride


class LearnableFilter2d(_BaseFixedFilter):
    def __init__(
        self,
        dim_in: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        filter_num: int = 1,
        include_bias: bool = True,
        reduce_layer: t.Optional[base.BaseComponent] = None,
        inverted_bias: bool = False,
    ):
        assert int(dim_in) > 0
        assert int(filter_num) > 0

        super(LearnableFilter2d, self).__init__(
            num_spatial_dims=2,
            trainable=True,
            kernel_size=kernel_size,
        )

        inverted_bias = bool(inverted_bias)

        std_reg = int(filter_num) if inverted_bias else int(dim_in)

        self.weights = base.Tensor.from_shape(
            (1, *self.kernel_size, int(dim_in), int(filter_num)),
            mode="uniform",
            std=("xavier", float(dim_in * np.prod(self.kernel_size))),
        )
        self.bias = base.Tensor()

        if include_bias:
            self.bias = base.Tensor.from_shape(
                (1, int(dim_in) if inverted_bias else int(filter_num)), mode="zeros"
            )
            self.parameters = (self.weights, self.bias)

        else:
            self.parameters = (self.weights,)

        if reduce_layer is None:
            reduce_layer = base.Sum(
                axis=(1, 2, 3),
                enforce_batch_dim=True,
                keepdims=False,
            )

        self.reduce_layer = reduce_layer
        self.mult = base.Multiply()
        self.add = base.Add()

        self.register_layers(self.reduce_layer, self.mult, self.add)

    def forward(self, X):
        if X.ndim != 5:
            X = np.expand_dims(X, -1)
            self._store_in_cache(True)

        else:
            self._store_in_cache(False)

        # X (shape): (batch, height, width, channels, 1)
        # W (shape): (1, height, width, channels, num_filters)

        out = self.mult(X, self.weights.values)
        out = self.reduce_layer(out)

        if self.bias.size:
            out = self.add(out, self.bias.values)

        return out

    def backward(self, dout):
        (squeeze_last_dim,) = self._pop_from_cache()

        if self.bias.size:
            dout, db = self.add.backward(dout)
            self.bias.update_grads(db)

        dout = self.reduce_layer.backward(dout)
        dX, dW = self.mult.backward(dout)

        self.weights.update_grads(dW)

        if squeeze_last_dim:
            dX = np.squeeze(dX, -1)

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

        self.filters = LearnableFilter2d(
            dim_in=self.channels_in,
            kernel_size=self.kernel_size,
            filter_num=self.channels_out,
            include_bias=include_bias,
        )

        self.register_layers(self.filters)

    def forward(self, X):
        input_shape = X.shape

        h_out_dim = self.calc_out_spatial_dim(X, 0)
        w_out_dim = self.calc_out_spatial_dim(X, 1)

        if self.use_padding:
            X = np.pad(X, pad_width=self.pad_widths, mode=self.padding_mode)

        input_shape_padded = X.shape

        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size
        batch_size, h_dim, w_dim, _ = X.shape

        out = np.empty(
            (batch_size, h_out_dim, w_out_dim, self.channels_out), dtype=float
        )

        for r, h_start in enumerate(range(0, h_dim - h_kernel + 1, h_stride)):
            h_end = h_start + h_kernel
            for c, w_start in enumerate(range(0, w_dim - w_kernel + 1, w_stride)):
                w_end = w_start + w_kernel

                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                out[:, r, c, :] = self.filters(X_slice)

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
        _, h_dout, w_dout, _ = dout.shape

        for r in reversed(range(h_dout)):
            h_start = r * h_stride
            h_end = h_start + h_kernel
            for c in reversed(range(w_dout)):
                w_start = c * w_stride
                w_end = w_start + w_kernel

                filter_grad = self.filters.backward(dout[:, r, c, :])
                dout_b[:, h_start:h_end, w_start:w_end, :] += filter_grad

        if self.use_padding:
            dout_b = self.crop_center(dout_b, input_shape)

        return dout_b


class ConvTranspose2d(_BaseConv):
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
        super(ConvTranspose2d, self).__init__(
            num_spatial_dims=2,
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding_type=padding_type,
            padding_mode=padding_mode,
            activation=activation,
        )

        self.filters = LearnableFilter2d(
            dim_in=self.channels_out,
            kernel_size=self.kernel_size,
            filter_num=self.channels_in,
            include_bias=include_bias,
            reduce_layer=base.Sum(axis=-1, keepdims=False),
            inverted_bias=True,
        )

        self.register_layers(self.filters)

    def forward(self, X):
        input_shape = X.shape

        if self.use_padding:
            X = np.pad(X, pad_width=self.pad_widths, mode=self.padding_mode)

        input_shape_padded = X.shape

        h_out_dim = self.calc_out_spatial_dim(X, 0, transpose=True)
        w_out_dim = self.calc_out_spatial_dim(X, 1, transpose=True)

        batch_dim, h_X, w_X, _ = X.shape

        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size

        out = np.zeros(
            (batch_dim, h_out_dim, w_out_dim, self.channels_out), dtype=float
        )

        for r in range(h_X):
            h_start = r * h_stride
            h_end = h_start + h_kernel

            for c in range(w_X):
                w_start = c * w_stride
                w_end = w_start + w_kernel

                X_slice = np.expand_dims(X[:, r, c, :], (1, 2, 3))
                filter_act = self.filters(X_slice)
                out[:, h_start:h_end, w_start:w_end, :] += filter_act

        if self.activation is not None:
            out = self.activation(out)

        self._store_in_cache(input_shape, input_shape_padded)

        return out

    def backward(self, dout):
        (input_shape, input_shape_padded) = self._pop_from_cache()

        if self.activation is not None:
            dout = self.activation.backward(dout)

        dout_b = np.empty(input_shape_padded, dtype=float)

        _, h_X, w_X, d_X = input_shape_padded
        batch_dim, h_dout, w_dout, _ = dout.shape
        h_stride, w_stride = self.stride
        h_kernel, w_kernel = self.kernel_size

        for r_inv, h_start in enumerate(
            reversed(range(0, h_dout - h_kernel + 1, h_stride))
        ):
            r = h_X - r_inv - 1
            h_end = h_start + h_kernel

            for c_inv, w_start in enumerate(
                reversed(range(0, w_dout - w_kernel + 1, w_stride))
            ):
                c = w_X - c_inv - 1
                w_end = w_start + w_kernel

                dout_slice = dout[:, h_start:h_end, w_start:w_end, :]

                filter_grads = self.filters.backward(dout_slice)
                filter_grads = np.squeeze(filter_grads)
                dout_b[:, r, c, :] = filter_grads

        if self.use_padding:
            dout_b = self.crop_center(dout_b, input_shape)

        return dout_b


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
        batch_size, h_dim, w_dim, _ = X.shape

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
        _, h_dout, w_dout, _ = dout.shape

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


class _BaseUpsample(base.BaseLayer):
    def __init__(
        self, num_spatial_dim: int, scale_factor: t.Union[float, t.Tuple[float, ...]]
    ):
        assert float(scale_factor) >= 1.0
        assert int(num_spatial_dim) > 0

        super(_BaseUpsample, self).__init__()

        self.num_spatial_dim = int(num_spatial_dim)

        scale_factor = _utils.replicate(scale_factor, self.num_spatial_dim)
        scale_factor = (0, *scale_factor, 0)

        self.scale_factor = tuple(scale_factor)
        self.scale_factor_ceil = tuple(np.ceil(scale_factor).astype(int, copy=False))
        self.scale_factor_rem = tuple(
            np.subtract(self.scale_factor_ceil, self.scale_factor)
        )


class NNUpsample(_BaseUpsample):
    def forward(self, X):
        out = X
        dim_adjusts = []
        all_rem_inds = []

        for axis in range(1, X.ndim - 1):
            out = np.repeat(out, repeats=self.scale_factor_ceil[axis], axis=axis)
            excess_to_remove = int(np.ceil(self.scale_factor_rem[axis] * X.shape[axis]))
            if excess_to_remove != 0:
                dim_adjusts.append((axis, excess_to_remove))

        for axis, excess_to_remove in dim_adjusts:
            rem_inds = np.fromiter(range(excess_to_remove), dtype=int)
            rem_inds = out.shape[axis] - 1 - rem_inds * self.scale_factor_ceil[axis]
            out = np.delete(out, rem_inds, axis=axis)
            all_rem_inds.append((axis, rem_inds))

        self._store_in_cache(all_rem_inds)

        return out

    def backward(self, dout):
        (all_rem_inds,) = self._pop_from_cache()
        ndim = len(dout.shape)
        axis_selector = [slice(None)] * ndim

        for axis, rem_inds in all_rem_inds:
            rem_inds -= np.fromiter(reversed(range(1, 1 + len(rem_inds))), dtype=int)
            dout = np.insert(dout, rem_inds, 0.0, axis=axis)

        for axis in range(1, ndim - 1):
            new_shape = list(dout.shape)
            n_reps = self.scale_factor_ceil[axis]
            new_axis_size = dout.shape[axis] // n_reps
            new_shape[axis] = new_axis_size

            dout_b = np.zeros(new_shape, dtype=float)

            for i, start in enumerate(range(0, dout.shape[axis], n_reps)):
                end = start + n_reps
                axis_selector[axis] = slice(start, end)
                dout_slice = dout[tuple(axis_selector)]
                axis_selector[axis] = [i]
                dout_b[tuple(axis_selector)] = np.sum(
                    dout_slice, axis=axis, keepdims=True
                )
                axis_selector[axis] = slice(None)

            dout = dout_b

        return dout
