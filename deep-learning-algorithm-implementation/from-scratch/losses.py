import typing as t

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
    def __init__(
        self, average: bool = True, with_logits: bool = False, eps: float = 1e-7
    ):
        super(BCELoss, self).__init__(average)
        self.sigmoid = modules.Sigmoid() if bool(with_logits) else None
        self.eps = float(eps)

    def __call__(self, y, y_preds):
        y = y.reshape(y_preds.shape)

        pos_inds = y >= 0.999

        if self.sigmoid is not None:
            y_preds = self.sigmoid.forward(y_preds)

        bce_loss = -float(
            np.sum(np.log(self.eps + y_preds[pos_inds]))
            + np.sum(np.log(self.eps + 1.0 - y_preds[~pos_inds]))
        )

        grads = (y_preds - y) / (self.eps + y_preds * (1.0 - y_preds))

        if self.average:
            bce_loss /= y.size
            grads /= y.size

        if self.sigmoid is not None:
            grads = self.sigmoid.backward(grads)

        return bce_loss, grads


class CrossEntropyLoss(_BaseLoss):
    def __init__(self, average: bool = True, ignore_index: int = -100):
        super(CrossEntropyLoss, self).__init__(average=average)
        self.ignore_index = int(ignore_index)

    def __call__(self, y, y_logits):
        y = y.reshape(-1, 1).astype(int, copy=False)

        assert y.size == y_logits.shape[0]

        ignored_inds = (y == self.ignore_index).ravel()

        grads = scipy.special.softmax(y_logits, axis=-1)
        grads[np.arange(y.size), y.ravel()] -= 1.0
        grads[ignored_inds, :] = 0.0

        log_probs = y_logits - scipy.special.logsumexp(y_logits, axis=-1, keepdims=True)
        ce_loss = np.take_along_axis(log_probs, y, axis=-1)
        ce_loss[ignored_inds] = 0.0
        ce_loss = -float(np.sum(ce_loss))

        if self.average:
            valid_tokens_num = int(np.sum(~ignored_inds)) + 1e-7
            ce_loss /= valid_tokens_num
            grads /= valid_tokens_num

        return ce_loss, grads


class AverageLosses:
    def __init__(
        self,
        criterions: t.Tuple[_BaseLoss, ...],
        weights: t.Optional[t.Tuple[float, ...]] = None,
        separated_y_true: bool = False,
        separated_y_preds: bool = False,
    ):
        n = len(criterions)

        if weights is None:
            weights = 1.0 / n

        self.weights = modules._utils.replicate(weights, n)
        self.criterions = tuple(criterions)

        self.separated_y_true = bool(separated_y_true)
        self.separated_y_preds = bool(separated_y_preds)

    def __len__(self):
        return len(self.criterions)

    def __repr__(self):
        strs = [f"Weighted average of {len(self)} loss functions:\n > Total Loss ="]
        for i in range(len(self)):
            criterion_name = type(self.criterions[i]).__name__
            weight = self.weights[i]
            strs.append(f"{'+ ' if i != 0 else ''}{weight:2.2f} * {criterion_name}")

        return " ".join(strs)

    def __call__(self, y, y_preds):
        loss = 0.0
        grads = []

        for i in np.arange(len(self)):
            criterion = self.criterions[i]
            weight = self.weights[i]

            y_true_cur = y[i] if self.separated_y_true else y
            y_preds_cur = y_preds[i] if self.separated_y_preds else y_preds

            cur_loss, cur_grads = criterion(y_true_cur, y_preds_cur)

            if grads is None:
                grads = np.zeros_like(cur_grads)

            loss += weight * cur_loss
            grads.append(weight * cur_grads)

        return loss, tuple(grads)
