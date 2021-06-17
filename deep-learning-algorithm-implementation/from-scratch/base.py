import numpy as np


class BaseModel:
    def __init__(self):
        self.frozen = False
        self.layers = tuple()

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

    def __call__(self, X):
        return self.forward(X)

    def train(self):
        self.frozen = False
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.frozen = True
        for layer in self.layers:
            layer.eval()
