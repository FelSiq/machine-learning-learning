import numpy as np

from . import base


class _BaseDropout(base.BaseLayer):
    def __init__(self, drop_prob: float, inplace: bool = False):
        assert 1.0 > float(drop_prob) >= 0.0
        super(_BaseDropout, self).__init__()
        self.drop_prob = float(drop_prob)
        self.keep_prob = 1.0 - self.drop_prob
        self.inplace = bool(inplace)


class Dropout(_BaseDropout):
    def forward(self, X):
        if self.frozen:
            out = X / self.keep_prob
            return out

        dropped_mask = np.random.random(X.shape) <= self.drop_prob

        if not self.inplace:
            X = np.copy(X)

        X[dropped_mask] = 0.0

        self._store_in_cache(dropped_mask)

        return X

    def backward(self, dout):
        (dropped_mask,) = self._pop_from_cache()

        if not self.inplace:
            dout = np.copy(dout)

        dout[dropped_mask] = 0.0

        return dout


class Dropout2d(_BaseDropout):
    def forward(self, X):
        if self.frozen:
            out = X / self.keep_prob
            return out

        num_channels = X.shape[-1]
        dropped_channel_mask = np.random.random(num_channels) <= self.drop_prob

        if not self.inplace:
            X = np.copy(X)

        X[..., dropped_channel_mask] = 0.0

        self._store_in_cache(dropped_channel_mask)

        return X

    def backward(self, dout):
        (dropped_channel_mask,) = self._pop_from_cache()

        if not self.inplace:
            dout = np.copy(dout)

        dout[..., dropped_channel_mask] = 0.0

        return dout
