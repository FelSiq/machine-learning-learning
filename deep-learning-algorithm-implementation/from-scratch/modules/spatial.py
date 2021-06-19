import typing as t

import numpy as np

from . import base
from . import _utils


class _BaseFilter(base.BaseLayer):
    def __init__(
        self,
        num_dims: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
    ):
        assert int(num_dims) > 0
        assert _utils.all_positive(kernel_size)
        assert _utils.all_positive(stride)

        self.stride = stride
        self.kernel_size = kernel_size


class _BaseConv(_BaseFilter):
    def __init__(
        self,
        num_dims: int,
        channels_in: int,
        channels_out: int,
        kernel_size: t.Union[int, t.Tuple[int, ...]],
        stride: t.Union[int, t.Tuple[int, ...]] = 1,
        padding: int = 0,
        padding_mode: str = "zeros",
    ):
        assert int(channels_in) > 0
        assert int(channels_out) > 0
        assert padding_mode in {"zeros", "reflect", "replicate", "circular"}

        super(_BaseConv, self).__init__(num_dims, kernel_size, stride)

        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.padding = int(padding)


class _BaseDropout(base.BaseLayer):
    pass


class Conv2d(_BaseConv):
    pass


class Dropout2d(_BaseDropout):
    pass


class MaxPool2d(_BaseFilter):
    pass


class AvgPool2d(_BaseFilter):
    pass
