import typing as t

import numpy as np

from . import base


class Standardization(base.BaseLayer):
    def __init__(
        self,
        axis: t.Optional[t.Tuple[int, ...]] = None,
        moving_avg_shape: t.Optional[t.Tuple[int, ...]] = None,
        momentum: float = 0.9,
        return_train_stats: bool = False,
        eps: float = 1e-6,
    ):
        assert float(eps) > 0.0

        super(Standardization, self).__init__()

        self.return_train_stats = bool(return_train_stats)
        self.eps = float(eps)

        self.std_and_avg = base.StandardDeviation(axis=axis, return_avg=True)
        self.axis = self.std_and_avg.axis
        self.divide = base.Divide()

        self.register_layers(self.std_and_avg, self.divide)

        self.moving_avg_stats = None

        if moving_avg_shape is not None:
            if not hasattr(moving_avg_shape, "__len__"):
                moving_avg_shape = (moving_avg_shape,)

            self.moving_avg_stats = base.MovingAverage(
                (2, *moving_avg_shape), momentum=momentum
            )

    def forward(self, X):
        if self.frozen and self.moving_avg_stats is not None:
            mov_avg, mov_std = self.moving_avg_stats
            X_norm = np.divide(X - mov_avg, mov_std + self.eps)
            return X_norm

        std, avg = self.std_and_avg(X)

        if self.moving_avg_stats is not None:
            std = std.reshape(self.moving_avg_stats.shape[1:])
            avg = avg.reshape(self.moving_avg_stats.shape[1:])

            self.moving_avg_stats.update([avg, std])

        X_norm = self.divide(X - avg, std + self.eps)

        if self.return_train_stats:
            return X_norm, avg, std

        return X_norm

    def backward(self, dout):
        dX_centered, d_std = self.divide.backward(dout)

        dX_a = dX_centered
        d_avg_extern = -dX_centered

        dX_b = self.std_and_avg.backward(d_std, d_avg_extern)

        dX = dX_a + dX_b

        return dX


class _BaseNorm(base.BaseLayer):
    def __init__(
        self,
        dim_in: int,
        scale_shape: t.Tuple[int, ...],
        affine: bool,
        standardization_axis: int,
        moving_avg_shape: t.Optional[t.Tuple[int, ...]] = None,
        momentum: float = 0.9,
        eps: float = 1e-6,
    ):
        assert int(dim_in) > 0

        super(_BaseNorm, self).__init__()

        self.dim_in = int(dim_in)
        self.affine = bool(affine)

        if self.affine:
            self.gamma = base.Tensor.from_shape(scale_shape, mode="constant", value=1)
            self.beta = base.Tensor.from_shape(scale_shape, mode="zeros")

            self.parameters = (
                self.gamma,
                self.beta,
            )

        self.standardization = Standardization(
            axis=standardization_axis,
            moving_avg_shape=moving_avg_shape,
            momentum=momentum,
            eps=eps,
        )

        self.multiply = base.Multiply()

        self.register_layers(self.multiply, self.standardization)

        self.standardization_axis = self.standardization.axis

    def forward(self, X):
        out = self.standardization(X)

        if self.affine:
            out = self.multiply(out, self.gamma.values) + self.beta.values

        return out

    def backward(self, dout):
        if self.affine:
            dbeta = np.sum(dout, axis=self.standardization_axis, keepdims=True)

            dout, dgamma = self.multiply.backward(dout)
            dgamma = np.sum(dgamma, axis=self.standardization_axis, keepdims=True)

            self.gamma.update_grads(dgamma)
            self.beta.update_grads(dbeta)

        dout = self.standardization.backward(dout)

        return dout


class BatchNorm1d(_BaseNorm):
    def __init__(
        self,
        dim_in: int,
        affine: bool = True,
        moving_avg_stats: bool = True,
        momentum: float = 0.9,
    ):
        super(BatchNorm1d, self).__init__(
            dim_in=dim_in,
            scale_shape=(1, dim_in),
            standardization_axis=0,
            moving_avg_shape=dim_in if moving_avg_stats else None,
            affine=affine,
            momentum=momentum,
        )


class BatchNorm2d(_BaseNorm):
    def __init__(
        self,
        dim_in: int,
        affine: bool = True,
        moving_avg_stats: bool = True,
        momentum: float = 0.9,
    ):
        super(BatchNorm2d, self).__init__(
            dim_in=dim_in,
            scale_shape=(1, 1, 1, dim_in),
            standardization_axis=(0, 1, 2),
            moving_avg_shape=(1, 1, 1, dim_in) if moving_avg_stats else None,
            affine=affine,
            momentum=momentum,
        )


class GroupNorm2d(_BaseNorm):
    def __init__(
        self,
        dim_in: int,
        num_groups: int,
        affine: bool = True,
        momentum: float = 0.9,
    ):
        assert int(num_groups) > 0

        super(GroupNorm2d, self).__init__(
            dim_in=dim_in,
            scale_shape=(1, 1, 1, dim_in, 1),
            standardization_axis=(1, 2, 3),
            moving_avg_shape=None,
            affine=affine,
            momentum=momentum,
        )

        self.num_groups = int(num_groups)

    def forward(self, X):
        inp_shape = X.shape
        (dim_b, dim_h, dim_w, dim_c) = inp_shape
        X = X.reshape(dim_b, dim_h, dim_w, self.num_groups, dim_c // self.num_groups)
        out = super(GroupNorm2d, self).forward(X)
        out = out.reshape(inp_shape)
        return out

    def backward(self, dout):
        dout_shape = dout.shape
        (dim_b, dim_h, dim_w, dim_c) = dout_shape
        dout = dout.reshape(
            dim_b, dim_h, dim_w, self.num_groups, dim_c // self.num_groups
        )
        dout = super(GroupNorm2d, self).backward(dout)
        dout.reshape(dout_shape)
        return dout


class InstanceNorm2d(_BaseNorm):
    pass


class LayerNorm2d(_BaseNorm):
    pass
