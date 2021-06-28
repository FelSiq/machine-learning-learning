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
        self.disable_when_frozen = bool(disable_when_frozen)
        self.noise_func = None

        self.noise_shape = noise_shape

        if self.noise_shape is not None:
            self.noise_shape = tuple(self.noise_shape)

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
        decay_std_max_iter: t.Optional[int] = None,
        decay_std_min: float = 0.0,
    ):
        assert float(std) >= 0.0
        assert 0.0 <= float(decay_std_min) <= float(std)

        super(AddNoiseGaussian, self).__init__(
            noise_shape=noise_shape, disable_when_frozen=disable_when_frozen
        )

        self.mean = float(mean)
        self.std = self.std_init = float(std)
        self.std_min = float(decay_std_min)

        self.D = decay_std_max_iter
        self.d = 0

        if self.D is not None:
            self.D = int(self.D)

        def noise_func(noise_shape):
            if self.D is not None and self.d < self.D:
                self.d += 1
                decay_factor = (self.D - self.d) / (self.D - self.std_init * self.d)
                self.std = self.std_min + (self.std_init - self.std_min) * decay_factor

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

        def noise_func(noise_shape):
            return np.random.uniform(low=self.low, high=self.high, size=noise_shape)

        self.noise_func = noise_func
