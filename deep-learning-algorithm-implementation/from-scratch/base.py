import numpy as np


class BaseModel:
    def _clip_grads(self, *grads):
        for grad in grads:
            np.clip(grad, -self.clip_grad_norm, self.clip_grad_norm, out=grad)

    def __call__(self, X):
        return self.forward(X)

    def train(self):
        for layer in self.layers:
            layer.frozen = False

    def eval(self):
        for layer in self.layers:
            layer.frozen = True
