import copy
import typing as t

import numpy as np

from . import _utils


class Tensor:
    def __init__(self, values: t.Optional[np.ndarray] = None):
        if values is None:
            values = np.empty(0, dtype=float)

        self.values = np.asfarray(values)
        self.grads = np.empty(0, dtype=float)
        self.frozen = False

    def step(self):
        if self.frozen or self.grads.size == 0:
            return

        self.values -= self.grads

    def update_and_step(self, grads):
        self.grads = grads
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

        constant = kwargs.get("value", 0.0) if mode == "constant" else 0.0
        return Tensor(np.full(shape, fill_value=constant, dtype=float))

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

    def eval(self):
        self.frozen = True

        for param in self.parameters:
            param.frozen = True

        for layer in self.layers:
            layer.eval()

    def register_layers(self, *layers):
        self.layers = (*self.layers, *layers)
        nested_parameters = []

        for layer in layers:
            nested_parameters.extend(layer.parameters)

        self.parameters = (*self.parameters, *nested_parameters)

    def copy(self):
        return copy.deepcopy(self)


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
    pass


class Identity(BaseLayer):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, X):
        return X

    def backward(self, dout):
        return dout


class Linear(BaseLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        include_bias: bool = True,
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        weight_init_std: t.Union[t.Tuple[str, str], float] = ("normal", "he"),
    ):
        assert dim_in > 0
        assert dim_out > 0

        super(Linear, self).__init__(trainable=True)

        if hasattr(weight_init_std, "__len__"):
            mode, std = weight_init_std

        else:
            mode, std = "normal", weight_init_std

        self.weights = Tensor.from_shape(
            shape=(dim_in, dim_out),
            mode=mode,
            std=std,
        )

        self.bias = Tensor()

        if include_bias:
            self.bias = Tensor.from_shape((dim_out), mode="zeros")
            self.parameters = (self.weights, self.bias)

        else:
            self.parameters = (self.weights,)

        self.activation = activation

        if self.activation is not None:
            self.register_layers(self.activation)

    def forward(self, X):
        out = X @ self.weights.values

        if self.bias.size:
            out += self.bias.values

        if self.activation is not None:
            out = self.activation(out)

        self._store_in_cache(X)

        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        if self.activation is not None:
            dout = self.activation.backward(dout)

        dW = X.T @ dout
        dX = dout @ self.weights.values.T

        self.weights.grads = dW

        if self.bias.size:
            db = np.sum(dout, axis=0)
            self.bias.grads = db

        return dX


class MultiLinear(BaseLayer):
    def __init__(
        self,
        dims_in: t.Sequence[int],
        dim_out: int,
        include_bias: bool = True,
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        weight_init_std: t.Union[t.Tuple[str, str], float] = ("normal", "he"),
    ):
        super(MultiLinear, self).__init__(trainable=True)

        self.activation = activation
        self.dim_out = int(dim_out)

        self.linear_transf = [
            Linear(
                dim_in,
                dim_out,
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


class Reshape(BaseLayer):
    def __init__(self, out_shape: t.Tuple[int, ...]):
        assert len(out_shape)
        super(Reshape, self).__init__()
        self.out_shape = tuple(out_shape)

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.reshape(-1, self.out_shape)

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        dout = dout.reshape(shape)
        return dout


class Flatten(BaseLayer):
    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.reshape(-1, np.prod(X.shape[1:]))

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        dout = dout.reshape(shape)
        return dout


class WeightedAverage(BaseLayer):
    def __call__(self, X, Y, W):
        return self.forward(X, Y, W)

    def forward(self, X, Y, W):
        out = W * X + (1.0 - W) * Y
        self._store_in_cache(X, Y, W)
        return out

    def backward(self, dout):
        X, Y, W = self._pop_from_cache()
        dX = W * dout
        dY = (1.0 - W) * dout
        dW = (X - Y) * dout
        return dX, dY, dW


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
        return dX, dY


class _BaseReduce(BaseLayer):
    def __init__(self, axes: t.Optional[t.Tuple[int, ...]] = None):
        super(_BaseReduce, self).__init__()
        self.axes = tuple(axes) if axes is not None else None


class Sum(_BaseReduce):
    def forward(self, X):
        self._store_in_cache(X.ndim)
        return np.sum(X, axis=self.axes)

    def backward(self, dout):
        X_ndim = self._pop_from_cache()

        if self.axes:
            dout = np.expand_dims(dout, self.axes)

        else:
            dout = np.expand_dims(dout, list(range(1, X_ndim)))

        return dout


class Average(_BaseReduce):
    def __init__(self, axes: t.Optional[t.Tuple[int, ...]] = None):
        super(Average, self).__init__()
        self.sum = Sum()
        self.register_layers(self.sum)

    def forward(self, X):
        self._store_in_cache(X.size)
        return self.sum(X) / X.size

    def backward(self, dout):
        (inp_size,) = self._pop_from_cache()
        dout = self.sum.backward(dout)
        return dout / inp_size
