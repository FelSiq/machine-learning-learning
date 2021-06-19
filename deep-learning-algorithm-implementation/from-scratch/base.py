import copy

import numpy as np

import modules


class BaseModel(modules.BaseComponent):
    def __init__(self):
        self.frozen = False
        self.layers = tuple()
        self.parameters = tuple()

    def _clip_grads(self, grads):
        if isinstance(grads, tuple):
            douts, param_grads = grads

        else:
            douts = grads
            param_grads = tuple()

        for grad in douts:
            np.clip(grad, -self.clip_grad_norm, self.clip_grad_norm, out=grad)

        for grad in param_grads:
            np.clip(grad, -self.clip_grad_norm, self.clip_grad_norm, out=grad)

    def copy(self):
        return copy.deepcopy(self)
