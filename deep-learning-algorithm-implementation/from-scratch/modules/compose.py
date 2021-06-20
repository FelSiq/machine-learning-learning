import typing as t

import numpy as np

from . import base


class Sequential(base.BaseLayer):
    def __init__(self, layers):
        super(Sequential, self).__init__(trainable=True)

        self.register_layers(*layers)

        self.param_nums = []  # type: t.List[int]

        for layer in self.layers:
            self.param_nums.append(len(layer.parameters))

        self.param_nums = tuple(self.param_nums)

    def forward(self, X):
        out = X

        for layer in self.layers:
            out = layer(out)

        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout
