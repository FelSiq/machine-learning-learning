import typing as t

import numpy as np

from . import base


class Sequential(base.BaseLayer):
    def __init__(self, layers):
        super(Sequential, self).__init__(trainable=True)
        self.layers = tuple(layers)

        self.parameters = []  # type: t.List[np.ndarray]
        self.param_nums = []  # type: t.List[int]

        for layer in self.layers:
            self.parameters.extend(layer.parameters)
            self.param_nums.append(len(layer.parameters))

        self.parameters = tuple(self.parameters)
        self.param_nums = tuple(self.param_nums)

    def forward(self, X):
        out = X

        for layer in self.layers:
            out = layer(out)

        return out

    def backward(self, dout):
        param_grads = []  # type: t.List[np.ndarray]

        for layer in reversed(self.layers):
            grads = layer.backward(dout)

            if not layer.trainable:
                dout = grads
                continue

            (dout,) = grads[0]
            cur_param_grads = grads[1]

            param_grads.extend(reversed(cur_param_grads))

        return (dout,), tuple(reversed(param_grads))

    def update(self, *args):
        if self.frozen:
            return

        start = 0

        for i, layer in enumerate(self.layers):
            if not layer.trainable:
                continue

            end = start + self.param_nums[i]
            cur_params = args[start:end]

            if len(cur_params) == 1:
                cur_params = (cur_params,)

            layer.update(*cur_params)
            start += self.param_nums[i]
