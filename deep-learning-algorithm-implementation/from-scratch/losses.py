import numpy as np
import scipy.special

import modules


class _BaseLoss:
    def __init__(self, average: bool = True):
        self.average = bool(average)

    def __call__(self, y, y_preds):
        raise NotImplementedError


class MSELoss(_BaseLoss):
    def __call__(self, y, y_preds):
        y = y.reshape(y_preds.shape)

        diff = y_preds - y

        mse = float(np.sum(diff * diff))
        grads = diff

        if self.average:
            mse /= y.size
            grads /= y.size

        return mse, grads


class BCELoss(_BaseLoss):
    def __init__(self, average: bool = True, with_logits: bool = False):
        super(BCELoss, self).__init__(average)
        self.sigmoid = modules.Sigmoid() if bool(with_logits) else None

    def __call__(self, y, y_preds):
        y = y.reshape(y_preds.shape)

        pos_inds = y >= 0.999

        if self.sigmoid is not None:
            y_preds = self.sigmoid.forward(y_preds)

        bce_loss = -float(
            np.sum(np.log(y_preds[pos_inds])) + np.sum(np.log(1.0 - y_preds[~pos_inds]))
        )

        grads = (y_preds - y) / (y_preds * (1.0 - y_preds))

        if self.average:
            bce_loss /= y.size
            grads /= y.size

        if self.sigmoid is not None:
            grads = self.sigmoid.backward(grads)

        return bce_loss, grads


class CrossEntropyLoss(_BaseLoss):
    def __init__(self, average: bool = True):
        super(CrossEntropyLoss, self).__init__(average)

    def __call__(self, y, y_logits):
        y = y.reshape(-1, 1).astype(int, copy=False)

        assert y.size == y_logits.shape[0]

        grads = scipy.special.softmax(y_logits, axis=-1)
        grads[np.arange(y.size), y.ravel()] -= 1.0

        log_probs = y_logits - scipy.special.logsumexp(y_logits, axis=-1, keepdims=True)
        ce_loss = -float(np.sum(np.take_along_axis(log_probs, y, axis=-1)))

        if self.average:
            ce_loss /= y.size
            grads /= y.size

        return ce_loss, grads
