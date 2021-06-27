import typing as t

import numpy as np

from . import base


class _AddNoiseBase(base.BaseLayer):
    def __init__(
        self,
        noise_shape: t.Optional[t.Tuple[int, ...]] = None,
        disable_when_frozen: bool = True,
    ):
        super(_AddNoiseBase, self).__init__()

        self.add = base.Add()
        self.register_layers(self.add)
        self.noise_shape = tuple(noise_shape)
        self.disable_when_frozen = bool(disable_when_frozen)
        self.noise_func = None

    def forward(self, X):
        if self.frozen and self.disable_when_frozen:
            return X

        noise_shape = self.noise_shape if self.noise_shape else X.shape
        noise = self.noise_func(noise_shape)
        out = self.add(X, noise)
        return out

    def backward(self, dout):
        dX, _ = self.add.backward(dout)
        return dX


class AddNoiseGaussian(_AddNoiseBase):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        noise_shape: t.Optional[t.Tuple[int, ...]] = None,
        disable_when_frozen: bool = True,
    ):
        assert float(std) >= 0.0

        super(AddNoiseGaussian, self).__init__(
            noise_shape=noise_shape, disable_when_frozen=disable_when_frozen
        )

        self.mean = float(mean)
        self.std = float(std)

        def noise_func(self, noise_shape):
            return np.random.normal(loc=self.mean, scale=self.std, size=noise_shape)

        self.noise_func = noise_func


class AddNoiseUniform(_AddNoiseBase):
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        noise_shape: t.Optional[t.Tuple[int, ...]] = None,
        disable_when_frozen: bool = True,
    ):
        assert float(high) >= float(low)

        super(AddNoiseUniform, self).__init__(
            noise_shape=noise_shape, disable_when_frozen=disable_when_frozen
        )

        self.low = float(low)
        self.high = float(high)

        def noise_func(self, noise_shape):
            return np.random.uniform(low=self.low, high=self.high, size=noise_shape)

        self.noise_func = noise_func
