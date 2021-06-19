import typing as t

import numpy as np


class _BaseLayer:
    def __init__(self, trainable: bool = False):
        self._cache = []
        self.trainable = trainable
        self.frozen = False
        self.parameters = tuple()
        self.layers = tuple()

    def _store_in_cache(self, *args):
        if self.frozen:
            return

        self._cache.append(args)

    def _pop_from_cache(self):
        return self._cache.pop()

    def forward(self, X):
        raise NotImplementedError

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dout):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError

    def clean_grad_cache(self):
        self._cache.clear()

    @property
    def has_stored_grads(self):
        return len(self._cache) > 0 or any(l.has_stored_grads for l in self.layers)

    def train(self):
        self.frozen = False
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.frozen = True
        for layer in self.layers:
            layer.eval()


class Linear(_BaseLayer):
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
        self.weights = he_init_coef * np.random.randn(dim_in, dim_out)
        self.bias = np.empty(0, dtype=float)

        if include_bias:
            self.bias = np.zeros(dim_out, dtype=float)
            self.parameters = (self.weights, self.bias)

        else:
            self.parameters = (self.weights,)

        self.activation = activation

    def forward(self, X):
        out = X @ self.weights

        if self.bias.size:
            out += self.bias

        if self.activation is not None:
            out = self.activation(out)

        self._store_in_cache(X)

        return out

    def backward(self, dout):
        (X,) = self._pop_from_cache()

        if self.activation is not None:
            dout = self.activation.backward(dout)

        dW = X.T @ dout
        dX = dout @ self.weights.T

        if self.bias.size:
            db = np.sum(dout, axis=0)
            return (dX,), (dW, db)

        return (dX,), (dW,)

    def update(self, *args):
        if self.frozen:
            return

        if self.bias.size:
            (dW, db) = args
            self.bias -= db
            return

        (dW,) = args
        self.weights -= dW


class MultiLinear(_BaseLayer):
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

        if inner_activations is None:
            inner_activations = [None] * len(dims_in)

        elif hasattr(inner_activations, "__len__"):
            assert len(inner_activations) == len(dims_in)

        else:
            inner_activations = [inner_activations] * len(dims_in)

        self.outer_activation = outer_activation

        self.layers = tuple(
            Linear(
                dim_in,
                dim_out,
                include_bias=(i == len(dims_in) and include_bias),
                activation=activation,
            )
            for i, (dim_in, activation) in enumerate(zip(dims_in, inner_activations), 1)
        )

        self.dim_out = int(dim_out)

        self.parameters = []

        for layer in self.layers:
            self.parameters.extend(layer.parameters)

        self.parameters = tuple(self.parameters)

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
        d_params = []  # type: t.List[np.ndarray]

        if self.outer_activation is not None:
            dout = self.outer_activation.backward(dout)

        for layer in self.layers:
            cur_d_outs, cur_d_params = layer.backward(dout)
            d_outs.extend(cur_d_outs)
            d_params.extend(cur_d_params)

        return tuple(d_outs), tuple(d_params)

    def update(self, *args):
        if self.frozen:
            return

        for layer, d_params in zip(self.layers[:-1], args[:-2]):
            layer.update(d_params)

        self.layers[-1].update(args[-2], args[-1])


class Reshape(_BaseLayer):
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


class WeightedAverage(_BaseLayer):
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
