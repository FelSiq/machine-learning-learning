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
        self.sub = base.Subtract()

        self.register_layers(self.std_and_avg, self.divide, self.sub)

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
            X_centered = self.sub(X, mov_avg)
            X_norm = np.divide(X_centered, mov_std + self.eps)
            return X_norm

        std, avg = self.std_and_avg(X)

        if self.moving_avg_stats is not None:
            std = std.reshape(self.moving_avg_stats.shape[1:])
            avg = avg.reshape(self.moving_avg_stats.shape[1:])

            self.moving_avg_stats.update([avg, std])

        X_centered = self.sub(X, avg)
        X_norm = self.divide(X_centered, std + self.eps)

        if self.return_train_stats:
            return X_norm, avg, std

        return X_norm

    def backward(self, dout):
        dX_centered, d_std = self.divide.backward(dout)
        dX_a, d_avg = self.sub.backward(dX_centered)
        dX_b = self.std_and_avg.backward(d_std, d_avg)
        dX = dX_a + dX_b
        return dX


class _BaseNorm(base.BaseLayer):
    def __init__(
        self,
        dim_in: int,
        affine_shape: t.Tuple[int, ...],
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
            self.gamma = base.Tensor.from_shape(affine_shape, mode="constant", value=1)
            self.beta = base.Tensor.from_shape(affine_shape, mode="zeros")

            self.parameters = (
                self.gamma,
                self.beta,
            )

            self._grad_sum_axis = tuple(i for i, s in enumerate(affine_shape) if s == 1)

        self.standardization = Standardization(
            axis=standardization_axis,
            moving_avg_shape=moving_avg_shape,
            momentum=momentum,
            eps=eps,
        )

        self.multiply = base.Multiply()
        self.add = base.Add()

        self.register_layers(self.multiply, self.standardization, self.add)

        self.standardization_axis = self.standardization.axis

    def forward(self, X):
        out = self.standardization(X)

        if self.affine:
            out = self.multiply(out, self.gamma.values)
            out = self.add(out, self.beta.values)

        return out

    def backward(self, dout):
        if self.affine:
            dout, dbeta = self.add.backward(dout)
            dbeta = np.sum(dout, axis=self._grad_sum_axis, keepdims=True)

            dout, dgamma = self.multiply.backward(dout)
            dgamma = np.sum(dgamma, axis=self._grad_sum_axis, keepdims=True)

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
            affine_shape=(1, dim_in),
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
            affine_shape=(1, 1, 1, dim_in),
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
        affine_shape: t.Optional[t.Tuple[int, ...]] = None,
    ):
        assert int(num_groups) > 0

        self.num_groups = int(num_groups)
        self.dim_per_group = int(dim_in) // self.num_groups

        affine_shape = (
            (1, *affine_shape)
            if affine_shape is not None
            else (1, 1, 1, self.num_groups, self.dim_per_group)
        )

        super(GroupNorm2d, self).__init__(
            dim_in=dim_in,
            affine_shape=affine_shape,
            standardization_axis=(1, 2, 3),
            moving_avg_shape=None,
            affine=affine,
        )

    def forward(self, X):
        inp_shape = X.shape
        X = X.reshape(*inp_shape[:-1], self.num_groups, self.dim_per_group)
        out = super(GroupNorm2d, self).forward(X)
        out = out.reshape(inp_shape)
        return out

    def backward(self, dout):
        shape = dout.shape
        dout = dout.reshape(*dout.shape[:-1], self.num_groups, self.dim_per_group)
        dout = super(GroupNorm2d, self).backward(dout)
        dout = dout.reshape(shape)
        return dout


class InstanceNorm2d(GroupNorm2d):
    def __init__(self, dim_in: int, affine: bool = True):
        super(InstanceNorm2d, self).__init__(
            dim_in=dim_in,
            num_groups=dim_in,
            affine=affine,
        )


class LayerNorm2d(GroupNorm2d):
    def __init__(self, input_shape: t.Tuple[int, ...], affine: bool = True):
        dim_in = input_shape[-1]
        dims_spatial = input_shape[:-1]

        super(LayerNorm2d, self).__init__(
            dim_in=dim_in,
            num_groups=1,
            affine=affine,
            affine_shape=(*dims_spatial, 1, dim_in),
        )
