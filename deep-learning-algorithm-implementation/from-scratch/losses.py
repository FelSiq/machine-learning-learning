import typing as t

import numpy as np
import scipy.special

import modules


class _BaseLoss:
    def __init__(self, average: bool = True):
        self.average = bool(average)


class _BasePairedLoss(_BaseLoss):
    def __call__(self, y, y_preds):
        raise NotImplementedError


class _BaseMatrixLoss(_BaseLoss):
    def __call__(self, sim_mat):
        raise NotImplementedError


class MSELoss(_BasePairedLoss):
    def __call__(self, y, y_preds):
        y = y.reshape(y_preds.shape)

        diff = y_preds - y

        mse = 0.5 * float(np.sum(diff * diff))
        grads = diff

        if self.average:
            mse /= y.size
            grads /= y.size

        return mse, grads


class BCELoss(_BasePairedLoss):
    def __init__(
        self, average: bool = True, with_logits: bool = False, eps: float = 1e-7
    ):
        assert float(eps) > 0.0
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
            + np.sum(np.log1p(-y_preds[~pos_inds] + self.eps))
        )

        grads = (y_preds - y) / (self.eps + y_preds * (1.0 - y_preds))

        if self.average:
            bce_loss /= y.size
            grads /= y.size

        if self.sigmoid is not None:
            grads = self.sigmoid.backward(grads)

        return bce_loss, grads


class CrossEntropyLoss(_BasePairedLoss):
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


class TripletLoss(_BaseMatrixLoss):
    def __init__(self, margin: float, average: bool = True):
        assert float(margin) >= 0.0
        super(TripletLoss, self).__init__(average=average)
        self.margin = float(margin)

        self.sum_negs = modules.Sum(axis=1, keepdims=False, enforce_batch_dim=False)
        self.max = modules.Max(axis=1, keepdims=False, enforce_batch_dim=False)
        self.sum_batch = modules.Sum(axis=0, keepdims=False, enforce_batch_dim=False)
        self.relu = modules.ReLU(inplace=True)
        self.scale = modules.ScaleByConstant()
        self.sub = modules.Subtract()

    def _neg_mean_forward(self, sim_mat_diag, sim_mat_off_diag):
        sim_sum_neg = self.sum_negs(sim_mat_off_diag)
        scale_const = 1.0 / (sim_sum_neg.size - 1.0)
        sim_mean_neg = self.scale(sim_sum_neg, scale_const)
        diff = self.sub(sim_mean_neg, sim_mat_diag)

        loss = self.relu(diff + self.margin)
        loss = self.sum_batch(loss)
        return float(loss)

    def _neg_mean_backward(self, dout):
        dout = self.relu.backward(dout)
        dsim_mean_neg, dsim_mat_diag = self.sub.backward(dout)
        dsim_sum_neg = self.scale.backward(dsim_mean_neg)
        dsim_mat_off_diag = self.sum_negs.backward(dsim_sum_neg)
        return dsim_mat_diag, dsim_mat_off_diag

    def _neg_closest_forward(self, sim_mat_diag, sim_mat_off_diag):
        mask_a = np.identity(sim_mat_diag.size).astype(bool, copy=False)
        mask_b = sim_mat_off_diag > sim_mat_diag.reshape(-1, 1)
        mask = np.logical_or(mask_a, mask_b)

        sim_mat_off_diag = np.copy(sim_mat_off_diag)
        sim_mat_off_diag[mask] = -2.0
        sim_closest_neg = self.max(sim_mat_off_diag)
        diff = self.sub(sim_closest_neg, sim_mat_diag)

        loss = self.relu(diff + self.margin)
        loss = self.sum_batch(loss)
        return float(loss)

    def _neg_closest_backward(self, dout):
        dout = self.relu.backward(dout)
        dsim_closes_neg, dsim_mat_diag = self.sub.backward(dout)
        dsim_mat_off_diag = self.max.backward(dsim_closes_neg)
        return dsim_mat_diag, dsim_mat_off_diag

    def __call__(self, sim_mat):
        sim_mat_diag = np.diag(sim_mat)
        sim_mat_off_diag = self.sub(sim_mat, np.diag(sim_mat_diag))

        loss_a = self._neg_mean_forward(sim_mat_diag, sim_mat_off_diag)
        loss_b = self._neg_closest_forward(sim_mat_diag, sim_mat_off_diag)

        loss = float(loss_a + loss_b)

        dsim_mat_diag_a, dsim_mat_off_diag_a = self._neg_closest_backward(1.0)
        dsim_mat_diag_b, dsim_mat_off_diag_b = self._neg_mean_backward(1.0)
        dsim_mat_off_diag = dsim_mat_off_diag_a + dsim_mat_off_diag_b
        dsim_mat_a, dsim_mat_diag_c = self.sub.backward(dsim_mat_off_diag)
        dsim_mat_diag_c = np.diag(dsim_mat_diag_c)
        dsim_mat_diag = dsim_mat_diag_a + dsim_mat_diag_b + dsim_mat_diag_c
        dsim_mat = dsim_mat_off_diag + np.diag(dsim_mat_diag)

        if self.average:
            n = sim_mat_diag.size
            loss /= n
            dsim_mat /= n

        return loss, dsim_mat


class AverageLosses:
    def __init__(
        self,
        criterions: t.Tuple[_BasePairedLoss, ...],
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


class _BaseRegularizer:
    def __init__(
        self, parameters: t.Sequence[modules.BaseComponent], loss: _BasePairedLoss
    ):
        self.parameters = parameters
        self.loss = loss


class RegularizerL1(_BaseRegularizer):
    def __init__(
        self,
        parameters: t.Sequence[modules.BaseComponent],
        loss: _BasePairedLoss,
        weight: float = 1.0,
    ):
        assert float(weight) >= 0.0
        super(RegularizerL1, self).__init__(parameters=parameters, loss=loss)
        self.weight = float(weight)

    def __call__(self, *args):
        loss, loss_grads = self.loss(*args)

        l1_loss = 0.0

        for param in self.parameters:
            l1_loss += float(np.sum(np.abs(param.values)))
            l1_loss_grads = self.weight * np.sign(param.values)
            param.update_grads(l1_loss_grads)

        l1_loss *= self.weight
        loss += l1_loss

        return loss, loss_grads


class RegularizerL2(_BaseRegularizer):
    def __init__(
        self,
        parameters: t.Sequence[modules.BaseComponent],
        loss: _BasePairedLoss,
        weight: float = 1.0,
    ):
        assert float(weight) >= 0.0
        super(RegularizerL2, self).__init__(parameters=parameters, loss=loss)
        self.weight = float(weight)

    def __call__(self, *args):
        loss, loss_grads = self.loss(*args)

        l2_loss = 0.0

        for param in self.parameters:
            l2_loss += float(np.sum(np.square(param.values)))
            l2_loss_grads = self.weight * param.values
            param.update_grads(l2_loss_grads)

        l2_loss *= 0.5 * self.weight
        loss += l2_loss

        return loss, loss_grads
