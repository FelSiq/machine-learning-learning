import numpy as np


class _BaseLoss:
    def __call__(self, y, y_preds):
        raise NotImplementedError


class MSELoss(_BaseLoss):
    def __call__(self, y, y_preds):
        diff = y_preds - y
        mse = np.sum(diff * diff) / y.size
        grads = diff / y.size
        return mse, grads
