import numpy as np

from . import base


class _BaseDropout(base.BaseLayer):
    def __init__(
        self, drop_prob: float, inplace: bool = False, disable_when_frozen: bool = True
    ):
        assert 1.0 > float(drop_prob) >= 0.0
        super(_BaseDropout, self).__init__()
        self.drop_prob = float(drop_prob)
        self.keep_prob = 1.0 - self.drop_prob
        self.inplace = bool(inplace)
        self.disable_when_frozen = bool(disable_when_frozen)


class Dropout(_BaseDropout):
    def forward(self, X):
        if self.frozen and self.disable_when_frozen:
            return X

        mask = np.random.random(X.shape) <= self.keep_prob
        mask = mask.astype(float, copy=False)
        mask /= self.keep_prob

        out = np.multiply(X, mask, out=X if self.inplace else None)

        self._store_in_cache(mask)

        return out

    def backward(self, dout):
        (mask,) = self._pop_from_cache()
        return mask * dout


class SpatialDropout(_BaseDropout):
    def forward(self, X):
        if self.frozen and self.disable_when_frozen:
            return X

        num_channels = X.shape[-1]
        dropped_channel_mask = np.random.random(num_channels) <= self.drop_prob

        if not self.inplace:
            X = np.copy(X)

        X[..., dropped_channel_mask] = 0.0
        X[..., ~dropped_channel_mask] /= self.keep_prob

        self._store_in_cache(dropped_channel_mask)

        return X

    def backward(self, dout):
        (dropped_channel_mask,) = self._pop_from_cache()

        if not self.inplace:
            dout = np.copy(dout)

        dout[..., dropped_channel_mask] = 0.0
        dout[..., ~dropped_channel_mask] /= self.keep_prob

        return dout
