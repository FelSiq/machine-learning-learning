import copy

import typing as t

import numpy as np


class Tensor:
    def __init__(self, values: t.Optional[np.ndarray] = None):
        if values is None:
            values = np.empty(0, dtype=float)

        self.values = np.asfarray(values)
        self.grads = np.empty(0, dtype=float)
        self.frozen = False

    def step(self):
        if self.frozen:
            return

        self.values -= self.grads

    def update_and_step(self, grads):
        self.grads = grads
        self.step()

    @property
    def size(self):
        return self.values.size


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
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.frozen = True
        for layer in self.layers:
            layer.eval()

    def register_layers(self, *layers):
        self.layers = tuple(layers)
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

    def clean_grad_cache(self):
        self._cache.clear()

    @property
    def has_stored_grads(self):
        return len(self._cache) > 0 or any(l.has_stored_grads for l in self.layers)

    def register_layers(self, *layers):
        if not self.trainable:
            assert all(map(lambda l: not l.trainable, layers))

        super(BaseLayer, self).register_layers(*layers)


class Linear(BaseLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        include_bias: bool = True,
        activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
    ):
        assert dim_in > 0
        assert dim_out > 0

        super(Linear, self).__init__(trainable=True)

        he_init_coef = np.sqrt(2.0 / dim_in)
        self.weights = Tensor(he_init_coef * np.random.randn(dim_in, dim_out))
        self.bias = Tensor()

        if include_bias:
            self.bias = Tensor(np.zeros(dim_out, dtype=float))
            self.parameters = (self.weights, self.bias)

        else:
            self.parameters = (self.weights,)

        self.activation = activation

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
        inner_activations: t.Optional[
            t.Union[t.Sequence, t.Callable[[np.ndarray], np.ndarray]]
        ] = None,
        outer_activation: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super(MultiLinear, self).__init__(trainable=True)

        if hasattr(inner_activations, "__len__"):
            assert len(inner_activations) == len(dims_in)

        else:
            inner_activations = [inner_activations] * len(dims_in)

        self.outer_activation = outer_activation
        self.dim_out = int(dim_out)

        layers = [
            Linear(
                dim_in,
                dim_out,
                include_bias=(i == len(dims_in) and include_bias),
                activation=activation,
            )
            for i, (dim_in, activation) in enumerate(zip(dims_in, inner_activations), 1)
        ]

        self.register_layers(*layers)

    def forward(self, *args):
        batch_size = args[0].shape[0]

        out = np.zeros((batch_size, self.dim_out), dtype=float)

        for layer, X in reversed(tuple(zip(self.layers, args))):
            out += layer(X)

        if self.outer_activation is not None:
            out = self.outer_activation(out)

        return out

    def __call__(self, *args):
        return self.forward(*args)

    def backward(self, dout):
        d_outs = []  # type: t.List[np.ndarray]

        if self.outer_activation is not None:
            dout = self.outer_activation.backward(dout)

        for layer in self.layers:
            cur_d_outs = layer.backward(dout)
            d_outs.append(cur_d_outs)

        return tuple(d_outs)


class Reshape(BaseLayer):
    def __init__(self, out_shape: t.Tuple[int, ...]):
        assert len(out_shape)
        super(Flatten, self).__init__()
        self.out_shape = tuple(out_shape)

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.reshape(self.out_shape)

    def backward(self, dout):
        (shape,) = self._pop_from_cache()
        dout = dout.reshape(shape)
        return dout


class Flatten(Reshape):
    def __init__(self):
        super(Flatten, self).__init__(out_shape=(-1,))

    def forward(self, X):
        self._store_in_cache(X.shape)
        return X.ravel()


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
