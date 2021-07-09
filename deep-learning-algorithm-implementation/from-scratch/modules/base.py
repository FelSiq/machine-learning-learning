import copy
import typing as t

import numpy as np

from . import _utils


class Tensor:
    def __init__(
        self, values: t.Optional[np.ndarray] = None, requires_grad: bool = True
    ):
        if values is None:
            values = np.empty(0, dtype=float)

        self.values = np.asfarray(values)

        self.grads = None

        if requires_grad:
            self.grads = np.zeros_like(self.values, dtype=float)

        self.frozen = False

    def step(self):
        if self.frozen or self.grads.size == 0:
            return

        self.values -= self.grads

    def zero_grad(self):
        self.grads *= 0.0

    def update_grads(self, grads):
        assert grads.shape == self.values.shape, (str(self), grads.shape)
        self.grads += grads

    def update_and_step(self, grads):
        self.update_grads(grads)
        self.step()

    @property
    def size(self):
        return self.values.size

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @staticmethod
    def from_shape(shape, mode: str = "normal", **kwargs):
        assert mode in {"normal", "uniform", "constant", "zeros"}

        if mode == "normal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            std = _utils.get_weight_init_dist_params(std, mode, shape)
            return Tensor(np.random.normal(mean, std, shape))

        if mode == "uniform":
            aux = kwargs.get("std", (None, None))
            init_type, dims = (aux, None) if isinstance(aux, str) else aux

            if init_type is not None:
                low, high = _utils.get_weight_init_dist_params(
                    init_type, mode, shape, dims
                )

            else:
                high = kwargs.get("high", 1.0)
                low = kwargs.get("low", -high)

            return Tensor(np.random.uniform(low, high, shape))

        constant = kwargs["value"] if mode == "constant" else 0.0
        return Tensor(np.full(shape, fill_value=constant, dtype=float))

    def init_weights(self, mode: str = "normal", **kwargs):
        self.values = Tensor.from_shape(self.shape, mode=mode, **kwargs).values
        return self

    def __repr__(self):
        tokens = []  # type: t.List[str]

        tokens.append(f"Tensor of shape {self.values.shape}")

        if self.frozen:
            tokens.append(" (frozen)")

        return "".join(tokens)


class BaseComponent:
    def __init__(self):
        self.frozen = False
        self.parameters = tuple()
        self.layers = tuple()

    def __iter__(self):
        return iter(self.layers)

    def forward(self, X):
        raise NotImplementedError

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dout):
        raise NotImplementedError

    def train(self):
        self.frozen = False

        for param in self.parameters:
            param.frozen = False

        for layer in self.layers:
            layer.train()

        return self

    def eval(self):
        self.frozen = True

        for param in self.parameters:
            param.frozen = True

        for layer in self.layers:
            layer.eval()

        return self

    def register_layers(self, *layers):
        self.layers = (*self.layers, *layers)
        nested_parameters = []

        for layer in layers:
            nested_parameters.extend(layer.parameters)

        self.parameters = (*self.parameters, *nested_parameters)

    def copy(self):
        return copy.deepcopy(self)

    @property
    def size(self):
        total_params = sum(p.size for p in self.parameters)
        return total_params

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def __repr__(self):
        strs = [f"{type(self).__name__} component"]

        if self.trainable:
            strs.append(
                f"with {len(self.parameters)} trainable "
                f"tensors (total of {self.size} parameters)"
            )

        if self.frozen:
            strs.append("(frozen)")

        return " ".join(strs)

    def init_weights(self, mode: str = "normal", **kwargs):
        for param in self.parameters:
            param.init_weights(mode=mode, **kwargs)

        return self


class MovingAverage(BaseComponent):
    def __init__(
        self,
        stat_shape: t.Tuple[int, ...],
        momentum: float = 0.9,
        init_const: float = 0.0,
    ):
        super(MovingAverage, self).__init__()

        assert 0.0 <= float(momentum) <= 1.0

        self.m = float(momentum)
        self.stat = np.full(stat_shape, fill_value=init_const, dtype=float)

    def update(self, new_stats):
        new_stats = np.asfarray(new_stats).reshape(self.stat.shape)
        self.stat *= self.m
        self.stat += (1.0 - self.m) * new_stats

    def __call__(self, new_stats):
        self.update(new_stats)

    def __iter__(self):
        return iter(self.stat)

    @property
    def shape(self):
        return self.stat.shape


class BaseLayer(BaseComponent):
    def __init__(self, trainable: bool = False):
        super(BaseLayer, self).__init__()
        self._cache = []
        self.trainable = bool(trainable)

    def _store_in_cache(self, *args):
        if self.frozen:
            return

        self._cache.append(args)

    def _pop_from_cache(self):
        return self._cache.pop()

    def clean_grad_cache(self, recurse: bool = True):
        self._cache.clear()

        if not recurse:
            return

        for layer in self.layers:
            layer.clean_grad_cache(recurse=recurse)

    @property
    def has_stored_grads(self):
        return len(self._cache) > 0 or any(l.has_stored_grads for l in self.layers)

    def register_layers(self, *layers):
        if not self.trainable:
            assert all(map(lambda l: not l.trainable, layers))

        super(BaseLayer, self).register_layers(*layers)


class BaseModel(BaseComponent):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.trainable = True


class Identity(BaseLayer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        return args if len(args) > 1 else args[0]

    def backward(self, *dout):
        return dout if len(dout) > 1 else dout[0]


class Linear(BaseLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_channels: int = 1,
        include_bias: bool = True,
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        weight_init_std: t.Union[t.Tuple[str, str], float] = ("normal", "he"),
    ):
        assert int(dim_in) > 0
        assert int(dim_out) > 0
        assert int(dim_channels) > 0

        super(Linear, self).__init__(trainable=True)

        if hasattr(weight_init_std, "__len__"):
            mode, std = weight_init_std

        else:
            mode, std = "normal", weight_init_std

        dim_channels = int(dim_channels)

        if dim_channels > 1:
            weight_shape = (dim_channels, dim_in, dim_out)
            bias_shape = (dim_channels, 1, dim_out)

        else:
            weight_shape = (dim_in, dim_out)
            bias_shape = (dim_out,)

        self.weights = Tensor.from_shape(
            shape=weight_shape,
            mode=mode,
            std=std,
        )

        self.bias = Tensor()

        if include_bias:
            self.bias = Tensor.from_shape(bias_shape, mode="zeros")
            self.parameters = (self.weights, self.bias)
            self.add = Add()
            self.register_layers(self.add)

        else:
            self.parameters = (self.weights,)

        self.activation = activation

        if self.activation is not None:
            self.register_layers(self.activation)

        self.matmul = Matmul()
        self.register_layers(self.matmul)

    def forward(self, X):
        out = self.matmul(X, self.weights.values)

        if self.bias.size:
            out = self.add(out, self.bias.values)

        if self.activation is not None:
            out = self.activation(out)

        self._store_in_cache(X)

        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        if self.activation is not None:
            dout = self.activation.backward(dout)

        if self.bias.size:
            dout, db = self.add.backward(dout)
            self.bias.update_grads(db)

        dX, dW = self.matmul.backward(dout)
        self.weights.update_grads(dW)

        return dX


class MultiLinear(BaseLayer):
    def __init__(
        self,
        dims_in: t.Sequence[int],
        dim_out: int,
        dim_channels: int = 1,
        include_bias: bool = True,
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        weight_init_std: t.Union[t.Tuple[str, str], float] = ("normal", "he"),
    ):
        super(MultiLinear, self).__init__(trainable=True)

        self.activation = activation
        self.dim_out = int(dim_out)

        self.linear_transf = [
            Linear(
                dim_in=dim_in,
                dim_out=dim_out,
                dim_channels=dim_channels,
                include_bias=(i == len(dims_in) and include_bias),
                activation=None,
                weight_init_std=weight_init_std,
            )
            for i, dim_in in enumerate(dims_in, 1)
        ]

        self.register_layers(*self.linear_transf)

        if self.activation is not None:
            self.register_layers(self.activation)

    def forward(self, *args):
        batch_size = args[0].shape[0]

        out = np.zeros((batch_size, self.dim_out), dtype=float)

        for layer, X in reversed(tuple(zip(self.linear_transf, args))):
            out += layer(X)

        if self.activation is not None:
            out = self.activation(out)

        return out

    def __call__(self, *args):
        return self.forward(*args)

    def backward(self, dout):
        d_outs = []  # type: t.List[np.ndarray]

        if self.activation is not None:
            dout = self.activation.backward(dout)

        for layer in self.linear_transf:
            d_outs.append(layer.backward(dout))

        return tuple(d_outs)


class Matmul(BaseLayer):
    def __init__(self, transpose_X: bool = False, transpose_Y: bool = False):
        super(Matmul, self).__init__()
        self.transpose_X = bool(transpose_X)
        self.transpose_Y = bool(transpose_Y)

    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        if self.transpose_X:
            X = np.swapaxes(X, -1, -2)

        if self.transpose_Y:
            Y = np.swapaxes(Y, -1, -2)

        self._store_in_cache(X, Y)

        out = np.matmul(X, Y)

        return out

    def backward(self, dout):
        (X, Y) = self._pop_from_cache()

        Xt = np.swapaxes(X, -1, -2)
        Yt = np.swapaxes(Y, -1, -2)

        dX = np.matmul(dout, Yt)
        dY = np.matmul(Xt, dout)

        dX = _utils.reduce_grad_broadcasting(dX, dout, X.shape)
        dY = _utils.reduce_grad_broadcasting(dY, dout, Y.shape)

        if self.transpose_Y:
            dY = np.swapaxes(dY, -1, -2)

        if self.transpose_X:
            dX = np.swapaxes(dX, -1, -2)

        return dX, dY


class Reshape(BaseLayer):
    def __init__(self, out_shape: t.Optional[t.Tuple[int, ...]] = None):
        assert out_shape is None or len(out_shape)

        super(Reshape, self).__init__()

        self.out_shape = None

        if out_shape is not None:
            self.out_shape = (-1, *out_shape)

    def __call__(self, X, out_shape=None):
        return self.forward(X, out_shape)

    def forward(self, X, out_shape: t.Optional[t.Tuple[int, ...]] = None):
        assert (out_shape is None) != (self.out_shape is None)

        if out_shape is None:
            out_shape = self.out_shape

        self._store_in_cache(X.shape)
        return X.reshape(out_shape)

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        dout = dout.reshape(shape)
        return dout


class Flatten(BaseLayer):
    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        dout = dout.reshape(shape)
        return dout


class Add(BaseLayer):
    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        self._store_in_cache(X.shape, Y.shape)
        return X + Y

    def backward(self, dout):
        (X_shape, Y_shape) = self._pop_from_cache()

        dX = np.ones(X_shape, dtype=float) * dout
        dY = np.ones(Y_shape, dtype=float) * dout

        dX = _utils.reduce_grad_broadcasting(dX, dout, X_shape)
        dY = _utils.reduce_grad_broadcasting(dY, dout, Y_shape)

        return dX, dY


class Subtract(BaseLayer):
    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        self._store_in_cache(X.shape, Y.shape)
        return X - Y

    def backward(self, dout):
        (X_shape, Y_shape) = self._pop_from_cache()

        dX = np.ones(X_shape, dtype=float) * dout
        dY = -np.ones(Y_shape, dtype=float) * dout

        dX = _utils.reduce_grad_broadcasting(dX, dout, X_shape)
        dY = _utils.reduce_grad_broadcasting(dY, dout, Y_shape)

        return dX, dY


class Multiply(BaseLayer):
    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        out = X * Y
        self._store_in_cache(X, Y)
        return out

    def backward(self, dout):
        (X, Y) = self._pop_from_cache()

        dX = Y * dout
        dY = X * dout

        dX = _utils.reduce_grad_broadcasting(dX, dout, X.shape)
        dY = _utils.reduce_grad_broadcasting(dY, dout, Y.shape)

        return dX, dY


class Divide(BaseLayer):
    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        out = X / Y
        self._store_in_cache(X, Y)
        return out

    def backward(self, dout):
        (X, Y) = self._pop_from_cache()

        dX = dout / Y
        dY = -X / np.square(Y) * dout

        dX = _utils.reduce_grad_broadcasting(dX, dout, X.shape)
        dY = _utils.reduce_grad_broadcasting(dY, dout, Y.shape)

        return dX, dY


class ScaleByConstant(BaseLayer):
    def __init__(self, constant: float):
        super(ScaleByConstant, self).__init__()
        self.constant = float(constant)

    def forward(self, X):
        return self.constant * X

    def backward(self, dout):
        return self.constant * dout


class WeightedAverage(BaseLayer):
    def __init__(self):
        super(WeightedAverage, self).__init__()
        self.add = Add()
        self.mult = Multiply()
        self.sub = Subtract()
        self.register_layers(self.add, self.mult, self.sub)

    def __call__(self, X, Y, W):
        return self.forward(X, Y, W)

    def forward(self, X, Y, W):
        out = self.add(self.mult(self.sub(X, Y), W), Y)
        return out

    def backward(self, dout):
        dout, dY_a = self.add.backward(dout)
        dout, dW = self.mult.backward(dout)
        dX, dY_b = self.sub.backward(dout)
        dY = dY_a + dY_b
        return dX, dY, dW


class _BaseReduce(BaseLayer):
    def __init__(
        self,
        axis: t.Optional[t.Tuple[int, ...]] = None,
        enforce_batch_dim: bool = True,
        keepdims: bool = True,
    ):
        super(_BaseReduce, self).__init__()

        if axis is not None and not hasattr(axis, "__len__"):
            self.axis = (int(axis),)

        else:
            self.axis = tuple(axis) if axis is not None else None

        self.enforce_batch_dim = bool(enforce_batch_dim)
        self.keepdims = bool(keepdims)


class _BaseAxisIndexGatherer(_BaseReduce):
    def __init__(
        self,
        indices_gather_func,
        axis: int = -1,
        enforce_batch_dim: bool = True,
        keepdims: bool = True,
    ):
        assert isinstance(axis, int)
        super(_BaseAxisIndexGatherer, self).__init__(
            axis=axis, enforce_batch_dim=enforce_batch_dim, keepdims=keepdims
        )
        self.axis = int(self.axis[0])
        self.indices_gather_func = indices_gather_func

    def forward(self, X):
        inds_chosen = np.argmax(X, axis=self.axis)
        inds_chosen = np.expand_dims(inds_chosen, self.axis)
        out = np.take_along_axis(X, inds_chosen, axis=self.axis)

        self._store_in_cache(inds_chosen, X.shape)

        if not self.keepdims:
            out = out.squeeze(self.axis)

        if self.enforce_batch_dim and out.ndim == 1:
            out = out.reshape(-1, 1)

        return out

    def backward(self, dout):
        (inds_chosen, input_shape) = self._pop_from_cache()

        if not self.keepdims:
            dout = np.expand_dims(dout, self.axis)

        aux_shape = [1] * len(input_shape)
        aux_shape[self.axis] = -1
        aux = np.arange(input_shape[self.axis]).reshape(aux_shape)

        dout_b = (inds_chosen == aux).astype(float, copy=False)
        dout_b *= dout

        return dout_b


class Max(_BaseAxisIndexGatherer):
    def __init__(
        self,
        axis: int = -1,
        enforce_batch_dim: bool = True,
        keepdims: bool = True,
    ):
        super(Max, self).__init__(
            indices_gather_func=np.argmax,
            axis=axis,
            enforce_batch_dim=enforce_batch_dim,
            keepdims=keepdims,
        )


class Min(_BaseAxisIndexGatherer):
    def __init__(
        self,
        axis: int = -1,
        enforce_batch_dim: bool = True,
        keepdims: bool = True,
    ):
        super(Min, self).__init__(
            indices_gather_func=np.argmin,
            axis=axis,
            enforce_batch_dim=enforce_batch_dim,
            keepdims=keepdims,
        )


class Sum(_BaseReduce):
    def forward(self, X):
        self._store_in_cache(X.shape)
        out = np.sum(X, axis=self.axis, keepdims=self.keepdims)

        if self.enforce_batch_dim and out.ndim == 1:
            out = out.reshape(-1, 1)

        return out

    def backward(self, dout):
        (inp_shape,) = self._pop_from_cache()

        if dout.ndim != len(inp_shape):
            dout = np.expand_dims(dout, self.axis)

        dout = np.ones(inp_shape, dtype=float) * dout

        return dout


class Average(Sum):
    def forward(self, X):
        if self.axis:
            inp_size = int(np.prod([X.shape[i] for i in self.axis]))

        else:
            inp_size = X.size

        avg = super(Average, self).forward(X) / inp_size
        self._store_in_cache(inp_size)

        return avg

    def backward(self, dout):
        (inp_size,) = self._pop_from_cache()
        dout = super(Average, self).backward(dout)
        dout = dout / inp_size
        return dout


class Power(BaseLayer):
    def __init__(self, power: float, eps: float = 1e-7):
        assert float(eps) > 0.0
        super(Power, self).__init__()
        self.power = float(power)
        self.eps = float(eps)

    def forward(self, X):
        X_pow_m1 = np.power(X + self.eps, self.power - 1)
        self._store_in_cache(X_pow_m1)
        return X_pow_m1 * X

    def backward(self, dout):
        (X_pow_m1,) = self._pop_from_cache()
        return self.power * X_pow_m1 * dout


class StandardDeviation(BaseLayer):
    def __init__(
        self,
        axis: t.Optional[t.Tuple[int, ...]] = None,
        return_avg: bool = False,
    ):
        super(StandardDeviation, self).__init__()

        self.return_avg = bool(return_avg)

        self.avg = Average(axis=axis, enforce_batch_dim=False, keepdims=True)
        self.square = Power(power=2)
        self.sqrt = Power(power=0.5)
        self.sub = Subtract()

        self.register_layers(self.avg, self.square, self.sqrt, self.sub)

        self.axis = self.avg.axis

    def forward(self, X):
        avg = self.avg(X)
        std = self.sqrt(self.avg(self.square(self.sub(X, avg))))

        if self.return_avg:
            return std, avg

        return std

    def backward(self, dout, d_avg_extern=None):
        dout = self.sqrt.backward(dout)
        dout = self.avg.backward(dout)
        dX_centered = self.square.backward(dout)
        dX_a, d_avg = self.sub.backward(dX_centered)

        if d_avg_extern is not None:
            d_avg += d_avg_extern

        dX_b = self.avg.backward(d_avg)

        dX = dX_a + dX_b

        return dX


class Exp(BaseLayer):
    def forward(self, X):
        out = np.exp(X)
        self._store_in_cache(out)
        return out

    def backward(self, dout):
        (out,) = self._pop_from_cache()
        return out * dout


class Log(BaseLayer):
    def forward(self, X):
        self._store_in_cache(X)
        return np.log(X)

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        return dout / X


class Flip(BaseLayer):
    def __init__(self, axis: t.Optional[t.Union[int, t.Tuple[int, ...]]] = None):
        super(Flip, self).__init__()

        if not hasattr(axis, "__len__"):
            axis = (axis,)

        self.axis = tuple(map(int, axis))

    def forward(self, X):
        return np.flip(X, axis=self.axis)

    def backward(self, dout):
        return np.flip(dout, axis=self.axis)


class _BaseCombineTensorLayer(BaseLayer):
    def __init__(
        self,
        axis: int,
        combine_fun: t.Callable,
        debug_ignore_axes: t.Optional[t.Tuple[int, ...]] = None,
    ):
        super(_BaseCombineTensorLayer, self).__init__()

        self.axis = int(axis)
        self.combine_fun = combine_fun

        self.debug_ignore_axes = None

        if debug_ignore_axes is not None:

            if not hasattr(debug_ignore_axes, "__len__"):
                debug_ignore_axes = (debug_ignore_axes,)

            self.debug_ignore_axes = tuple(debug_ignore_axes)

    def __call__(self, *tensors):
        return self.forward(*tensors)

    def forward(self, *tensors):
        self._store_in_cache(tuple(ts.shape for ts in tensors))
        tensors = np.broadcast_arrays(*tensors)
        return self.combine_fun(tensors, axis=self.axis)

    def backward(self, dout):
        raise NotImplementedError


class Stack(_BaseCombineTensorLayer):
    def __init__(
        self,
        axis: int,
        debug_ignore_axes: t.Optional[t.Tuple[int, ...]] = None,
    ):
        super(Stack, self).__init__(
            axis=axis, combine_fun=np.stack, debug_ignore_axes=debug_ignore_axes
        )

    def backward(self, dout):
        (tensor_shapes,) = self._pop_from_cache()
        num_tensors = len(tensor_shapes)

        douts = np.split(dout, num_tensors, axis=self.axis)
        douts = tuple(
            _utils.reduce_grad_broadcasting(
                np.squeeze(dX, axis=self.axis),
                dout,
                X_shape,
                debug_ignore_axes=self.debug_ignore_axes,
            )
            for dX, X_shape in zip(dtensors, tensor_shapes)
        )

        return douts


class Concatenate(_BaseCombineTensorLayer):
    def __init__(
        self,
        axis: int,
        debug_ignore_axes: t.Optional[t.Tuple[int, ...]] = None,
    ):
        super(Concatenate, self).__init__(
            axis=axis, combine_fun=np.concatenate, debug_ignore_axes=debug_ignore_axes
        )
        self.axis = int(axis)

    def backward(self, dout):
        (tensor_shapes,) = self._pop_from_cache()

        tensor_lens = tuple(ts[self.axis] for ts in tensor_shapes[1:])
        split_inds = np.cumsum(tensor_lens)

        dtensors = np.array_split(dout, split_inds, axis=self.axis)

        douts = tuple(
            _utils.reduce_grad_broadcasting(
                dX, dout, X_shape, debug_ignore_axes=self.debug_ignore_axes
            )
            for dX, X_shape in zip(dtensors, tensor_shapes)
        )

        return douts


class Split(BaseLayer):
    def __init__(self, num_slices: int, axis: int):
        assert int(num_slices) > 1
        super(Split, self).__init__()
        self.num_slices = int(num_slices)
        self.axis = int(axis)

    def forward(self, X):
        return np.split(X, self.num_slices, axis=self.axis)

    def backward(self, dout):
        return np.concatenate(dout, axis=self.axis)


class CollapseAdjacentAxes(BaseLayer):
    def __init__(self, axis_first: int, axis_last: int):
        axis_first = int(axis_first)
        axis_last = int(axis_last)

        is_nonneg_first = bool(np.heaviside(axis_first, 1))
        is_nonneg_last = bool(np.heaviside(axis_first, 1))

        if is_nonneg_first == is_nonneg_last:
            assert 1 <= axis_last - axis_first

        else:
            is_neg_last = not is_nonneg_last
            assert is_nonneg_first and is_neg_last

        super(CollapseAdjacentAxes, self).__init__()

        self.axis_first = axis_first
        self.axis_last = axis_last

    def forward(self, X):
        self._store_in_cache(X.shape)
        shape_bef, shape_middle, shape_after = np.array_split(
            X.shape, (self.axis_first, self.axis_last + 1)
        )
        new_shape = (*shape_bef, int(np.prod(shape_middle)), *shape_after)
        out = X.reshape(new_shape)
        return out

    def backward(self, dout):
        (input_shape,) = self._pop_from_cache()
        return dout.reshape(input_shape)


class PermuteAxes(BaseLayer):
    def __init__(self, permutation: t.Tuple[int, ...]):
        super(PermuteAxes, self).__init__()
        self.permutation = tuple(permutation)
        self.inv_permutation = tuple(np.argsort(self.permutation))

    def forward(self, X):
        return np.transpose(X, self.permutation)

    def backward(self, dout):
        return np.transpose(dout, self.inv_permutation)


class Abs(BaseLayer):
    def forward(self, X):
        self._store_in_cache(X)
        return np.abs(X)

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        return np.sign(X) * dout


class CrossEntropy(BaseLayer):
    def __init__(self, axis: t.Union[int, t.Tuple[int, ...]]):
        super(CrossEntropy, self).__init__()
        self.neg = ScaleByConstant(-1.0)
        self.log = Log()
        self.sum = Sum(axis=axis)
        self.mult = Multiply()

        self.register_layers(self.neg, self.log, self.sum, self.mult)

    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        Y_log = self.log(Y)
        out = self.neg(self.sum(self.multiply(X, Y_log)))
        return out

    def backward(self, dout):
        dout = self.neg.backward(dout)
        dout = self.sum.backward(dout)
        dX, dY_log = self.multiply.backward(dout)
        dY = self.log.backward(dY_log)
        return dX, dY


class Entropy(BaseLayer):
    def __init__(self, axis: t.Union[int, t.Tuple[int, ...]]):
        super(Entropy, self).__init__()
        self.ce = CrossEntropy(axis=axis)
        self.register_layers(self.ce)

    def forward(self, X):
        return self.ce(X, X)

    def backward(self, dout):
        dX_a, dX_b = self.ce.backward(dout)
        dX = dX_a + dX_b
        return dX


class NormP(BaseLayer):
    def __init__(
        self,
        p: t.Union[int, float],
        axis: t.Union[int, t.Tuple[int, ...]] = -1,
        root: bool = True,
    ):
        super(NormP, self).__init__()

        self.p = float(p)
        self.not_l1_norm = not np.isclose(1.0, self.p)
        self.use_abs = not (np.isclose(int(p), p) and p % 2 == 0)

        self.root = bool(root)

        self.sum = Sum(axis=axis)
        self.abs = Abs()

        if self.not_l1_norm:
            self.power = Power(power=p)
            self.power_inv = Power(power=1.0 / p)

        else:
            self.power = self.power_inv = Identity()

        self.register_layers(self.sum, self.abs, self.power, self.power_inv)

    def forward(self, X):
        out = self.abs(X) if self.use_abs else X
        out = self.power(out)
        out = self.sum(out)

        if self.root:
            out = self.power_inv(out)

        return out

    def backward(self, dout):
        if self.root:
            dout = self.power_inv.backward(dout)

        dout = self.sum.backward(dout)
        dout = self.power.backward(dout)
        dX = self.abs.backward(dout) if self.use_abs else dout
        return dX


class NormL2(NormP):
    def __init__(self, axis: t.Union[int, t.Tuple[int, ...]] = -1, root: bool = True):
        super(NormL2, self).__init__(p=2, axis=axis, root=root)


class NormL1(NormP):
    def __init__(self, axis: t.Union[int, t.Tuple[int, ...]] = -1):
        super(NormL2, self).__init__(p=1, axis=axis)


class NormalizeVector(BaseLayer):
    def __init__(
        self, p: t.Union[int, float] = 2, axis: t.Union[int, t.Tuple[int, ...]] = -1
    ):
        super(NormalizeVector, self).__init__()
        self.norm_p = NormP(p=p, axis=axis)
        self.div = Divide()
        self.register_layers(self.norm_p, self.div)

    def forward(self, X):
        X_norm = self.norm_p(X)
        out = self.div(X, X_norm)
        return out

    def backward(self, dout):
        dX_a, dX_norm = self.div.backward(dout)
        dX_b = self.norm_p.backward(dX_norm)
        dX = dX_a + dX_b
        return dX


class Cos(BaseLayer):
    def forward(self, X):
        self._store_in_cache(X)
        return np.cos(X)

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        return -np.sin(X) * dout


class Sin(BaseLayer):
    def forward(self, X):
        self._store_in_cache(X)
        return np.sin(X)

    def backward(self, dout):
        (X,) = self._pop_from_cache()
        return np.cos(X) * dout
