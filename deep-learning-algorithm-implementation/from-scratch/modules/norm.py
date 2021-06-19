import numpy as np

from . import base


class _BaseNorm(base.BaseLayer):
    pass


class BatchNorm2d(_BaseNorm):
    pass


class InstanceNorm2d(_BaseNorm):
    pass


class LayerNorm2d(_BaseNorm):
    pass


class GroupNorm2d(_BaseNorm):
    pass
