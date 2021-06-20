import typing as t

import numpy as np


class _BaseOptim:
    def __init__(self, parameters, learning_rate: float, clip_grad_val: float = 1.0):
        assert float(learning_rate) > 0.0
        assert float(clip_grad_val) > 0.0

        self.learning_rate = float(learning_rate)
        self.bias_correction = False
        self.register_layer(parameters)
        self._clip_grad_val = float(clip_grad_val)

    def register_layer(self, params):
        self.parameters = tuple(params)

    def step(self, layer_id: int, *grads):
        raise NotImplementedError

    @staticmethod
    def _unpack(vals):
        return vals[0] if len(vals) == 1 else vals

    def _correct_bias(self, it: int, m: float, *steps):
        mom_it = m ** it

        if not self.bias_correction or mom_it < 1e-3:
            return self._unpack(steps)

        unbiased = []

        for step in steps:
            unbiased.append(step / (1.0 - mom_it))

        return self._unpack(unbiased)

    def clip_grads_val(self):
        v = self._clip_grad_val

        for param in self.parameters:
            np.clip(param.grads, -v, v, out=param.grads)


class SGD(_BaseOptim):
    """Vanilla SGD.

    All parameters have the same learning rate.
    """

    def step(self):
        for i, param in enumerate(self.parameters):
            cur_steps = self.learning_rate * param.grads
            param.update_and_step(cur_steps)


class Momentum(_BaseOptim):
    """SGD with Momentum.

    Solves the difficulty of Vanilla SGD with surfaces that are too much
    steeper on only some directions than others. Since this algorithms adds
    a moving average to the model state position step, directions with
    high variability of direction cancel theirs own steps in average, while
    positions with low variability of direction builds up velocity and, hence,
    moves faster to the local minima.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float,
        first_momentum: float,
        bias_correction: bool = True,
        clip_grad_val: float = 1.0,
    ):
        assert 1.0 > float(first_momentum) > 0.0

        self.first_momentum = float(first_momentum)
        self.bias_correction = bias_correction
        self.fst_mom_mov_avg = []
        self.iterations = 0

        super(Momentum, self).__init__(
            parameters=parameters,
            learning_rate=learning_rate,
            clip_grad_val=clip_grad_val,
        )

    def register_layer(self, params):
        super(Momentum, self).register_layer(params)

        for param in params:
            self.fst_mom_mov_avg.append(np.zeros_like(param.values))

    def step(self):
        m = self.first_momentum

        for i, param in enumerate(self.parameters):
            prev_fst_mma = self.fst_mom_mov_avg[i]
            cur_fst_mma = m * prev_fst_mma + (1.0 - m) * param.grads

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            self.fst_mom_mov_avg[i] = cur_fst_mma

            cur_fst_mma = self._correct_bias(it, m, cur_fst_mma)
            cur_steps = self.learning_rate * cur_fst_mma

            param.update_and_step(cur_steps)

        self.iterations += 1


class NesterovMomentum(Momentum):
    """Nesterov Accelerated Gradient (NAG).

    This optimizer modifies the vanilla (SGD+)Momentum algorithm with a
    'lookahead' of the gradients of the approximated next position
    considering the current state velocity and, hence, this optimizer
    builds up momentum in a informed way, and not completely blind like
    the vanilla Momentum algorithm.
    """

    def step(self):
        m = self.first_momentum

        for i, param in enumerate(self.parameters):
            prev_fst_mma = self.fst_mom_mov_avg[i]
            cur_fst_mma = m * prev_fst_mma + (1.0 - m) * param.grads

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            self.fst_mom_mov_avg[i] = cur_fst_mma

            cur_fst_mma, prev_fst_mma = self._correct_bias(
                it, m, cur_fst_mma, prev_fst_mma
            )
            cur_steps = self.learning_rate * (
                (1.0 + m) * cur_fst_mma - m * prev_fst_mma
            )

            param.update_and_step(cur_steps)

        self.iterations += 1


class Adagrad(_BaseOptim):
    """Adagrad: Adaptive Gradient Optimizer.

    Not recommended since it is too aggressive while reducing the learning
    rates; the reduction is monotonic decreasing, and this may cause the
    learning to stop way too soon.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float,
        eps: float = 1e-8,
        clip_grad_val: float = 1.0,
    ):
        assert float(eps) > 0.0

        self.eps = float(eps)
        self.cum_sec_momentum = []

        super(Adagrad, self).__init__(
            parameters=parameters,
            learning_rate=learning_rate,
            clip_grad_val=clip_grad_val,
        )

    def register_layer(self, params):
        super(Adagrad, self).register_layer(params)

        for param in params:
            self.cum_sec_momentum.append(np.zeros_like(param.values))

    def step(self):
        for i, param in enumerate(self.parameters):
            prev_cum_sec_mom = self.cum_sec_momentum[i]
            cur_cum_sec_mom = prev_cum_sec_mom + np.square(param)

            self.cum_sec_momentum[i] = cur_cum_sec_mom

            cur_lr = self.learning_rate / np.sqrt(cur_cum_sec_mom + self.eps)
            cur_steps = cur_lr * param.grads

            param.update_and_step(cur_steps)


class Adadelta(_BaseOptim):
    """Adadelta optimization algorithm.

    Solves the problem of the aggressiveness of Adagrad, while preserving the
    adaptive learning rate per parameter.

    It is very similar to RMSProp optimizer, but was invented separately.

    The peculiar characteristic of Adadelta is that it does not depends on the
    learning rate hyperparameter.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float = 1.0,
        second_momentum: float = 0.9,
        eps: float = 1e-6,
        clip_grad_val: float = 1.0,
    ):
        assert float(eps) > 0.0
        assert 1.0 > float(second_momentum) > 0.0

        self.eps = float(eps)
        self.second_momentum = float(second_momentum)
        self.mv_avg_second_mom_grads = []
        self.mv_avg_second_mom_params = []

        super(Adadelta, self).__init__(
            parameters=parameters,
            learning_rate=learning_rate,
            clip_grad_val=clip_grad_val,
        )

    def register_layer(self, params):
        super(Adadelta, self).register_layer(params)

        for param in params:
            weights_grads, weights_params = np.zeros(
                (2, *param.values.shape), dtype=float
            )
            mv_avg_second_mom_grads.append(weights_grads)
            mv_avg_second_mom_params.append(weights_params)

    def step(self):
        m = self.second_momentum
        eps = self.eps

        for i, param in enumerate(self.parameters):
            prev_mov_avg_params = self.mv_avg_second_mom_params[i]
            prev_mov_avg_params = self.mv_avg_second_mom_params[i]

            cur_mov_avg_params = m * prev_mov_avg_params + (1.0 - m) * np.square(
                param.grads
            )

            # Note: both 'eps', in the numerator and denominator, are fundamental to
            # the algorithm works properly. The 'eps' on the denominator prevents division
            # by zero, while the 'eps' on the numerator starts the recursive moving
            # average as the value of the first paramient instead of just zeros,
            # which is necessary to actually start the model training.
            cur_lr = np.sqrt((prev_mov_avg_params + eps) / (cur_mov_avg_params + eps))
            cur_steps = cur_lr * param.grads

            cur_mov_avg_params = m * prev_mov_avg_params + (1.0 - m) * np.square(
                cur_steps
            )

            self.mv_avg_second_mom_params[i] = cur_mov_avg_params
            self.mv_avg_second_mom_params[i] = cur_mov_avg_params

            param.update_and_step(cur_steps)


class RMSProp(Adagrad):
    """RMSProp: Root Mean Square Propagation.

    Like Adadelta, it was invented to mitigate the aggressiveness of the
    Adagrad algorithm. Although they were invented separately, RMSProp
    actually uses the same strategy from Adadelta: instead of keeping the
    sum of squares of all gradients, RMSProp (and Adadelta) uses a moving
    average to store the square of the gradients; in fact, Adadelta is
    just RMSProp with one step further of modification in the parameter
    step rule.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float,
        second_momentum: float = 0.9,
        eps: float = 1e-8,
        clip_grad_val: float = 1.0,
    ):
        assert 1.0 > float(second_momentum) > 0.0

        self.second_momentum = float(second_momentum)
        self.sec_mom_mov_avg = []

        super(RMSProp, self).__init__(
            parameters=parameters,
            learning_rate=learning_rate,
            eps=eps,
            clip_grad_val=clip_grad_val,
        )

    def register_layer(self, params):
        super(RMSProp, self).register_layer(params)

        for param in params:
            self.sec_mom_mov_avg.append(np.zeros_like(param.values))

    def step(self):
        m = self.second_momentum

        for i, param in enumerate(self.parameters):
            prev_sec_mom_mov_avg = self.sec_mom_mov_avg[i]
            cur_sec_mom_mov_avg = m * prev_sec_mom_mov_avg + (1.0 - m) * np.square(
                param.grads
            )

            self.sec_mom_mov_avg[i] = cur_sec_mom_mov_avg

            cur_lr = self.learning_rate / np.sqrt(cur_sec_mom_mov_avg + self.eps)
            cur_steps = cur_lr * param.grads

            param.update_and_step(cur_steps)


class Adam(Momentum, RMSProp):
    """Adam: Adaptive Moment Estimation.

    Combines RMSProp with Momentum.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float,
        first_momentum: float = 0.9,
        second_momentum: float = 0.999,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        eps: float = 1e-8,
        weight_decay_ignore_ndims: t.Tuple[int] = (1,),
        clip_grad_val: float = 1.0,
    ):
        assert float(weight_decay) >= 0.0

        self.amsgrad = bool(amsgrad)
        self.weight_decay = float(weight_decay)
        self.weight_decay_ignore_ndims = frozenset(weight_decay_ignore_ndims)

        self.bias_correction = True
        self.params_should_decay = []

        Momentum.__init__(
            self,
            parameters,
            learning_rate=learning_rate,
            first_momentum=first_momentum,
            clip_grad_val=clip_grad_val,
        )

        RMSProp.__init__(
            self,
            parameters,
            learning_rate=learning_rate,
            second_momentum=second_momentum,
            eps=eps,
            clip_grad_val=clip_grad_val,
        )

    def register_layer(self, params):
        Momentum.register_layer(self, params)
        RMSProp.register_layer(self, params)

        for param in params:
            self.params_should_decay.append(
                param.values.ndim not in self.weight_decay_ignore_ndims
            )

    def step(self):
        m1 = self.first_momentum
        m2 = self.second_momentum
        it = self.iterations

        for i, param in enumerate(self.parameters):
            prev_fst_mma = self.fst_mom_mov_avg[i]
            prev_sec_mma = self.sec_mom_mov_avg[i]

            cur_fst_mma = m1 * prev_fst_mma + (1.0 - m1) * param.grads
            cur_sec_mma = m2 * prev_sec_mma + (1.0 - m2) * np.square(param.grads)

            if self.amsgrad:
                cur_sec_mma = np.maximum(prev_sec_mma, cur_sec_mma)

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            self.fst_mom_mov_avg[i] = cur_fst_mma
            self.sec_mom_mov_avg[i] = cur_sec_mma

            cur_fst_mma = self._correct_bias(it, m1, cur_fst_mma)
            cur_sec_mma = self._correct_bias(it, m2, cur_sec_mma)

            cur_lr = self.learning_rate / np.sqrt(cur_sec_mma + self.eps)

            decay_factor = self.weight_decay if self.params_should_decay[i] else 0.0
            cur_steps = cur_lr * (
                cur_fst_mma + decay_factor * self.parameters[i].values
            )

            param.update_and_step(cur_steps)

        self.iterations += 1


class Adamax(Adam):
    """Adam with second moment generalized to infinity norm.

    The main changes are listed below:
    - The second moment is not an moving average anymore; instead, it is
      an infinity norm between the scaled previous second moment and the
      absolute value of the current gradients.
    - Theres no need for bias correction onto the second moment anymore.
    - I don't know if it makes sense to combine Adamax with AMSgrad.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float,
        first_momentum: float = 0.9,
        second_momentum: float = 0.999,
        eps: float = 1e-8,
        clip_grad_val: float = 1.0,
    ):
        super(Adamax, self).__init__(
            parameters,
            learning_rate=learning_rate,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            eps=eps,
            amsgrad=False,
            clip_grad_val=clip_grad_val,
        )

    def step(self):
        m1 = self.first_momentum
        m2 = self.second_momentum
        it = self.iterations

        for i, param in enumerate(self.parameters):
            prev_fst_mma = self.fst_mom_mov_avg[i]
            prev_sec_mma = self.sec_mom_mov_avg[i]

            cur_fst_mma = m1 * prev_fst_mma + (1.0 - m1) * param.grads
            cur_sec_mma = np.maximum(m2 * prev_sec_mma, np.abs(param.grads))

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            self.fst_mom_mov_avg[i] = cur_fst_mma
            self.sec_mom_mov_avg[i] = cur_sec_mma

            cur_fst_mma = self._correct_bias(it, m1, cur_fst_mma)

            cur_lr = self.learning_rate / (cur_sec_mma + self.eps)
            decay_factor = self.weight_decay if self.params_should_decay[i] else 0.0
            cur_steps = cur_lr * (
                cur_fst_mma + decay_factor * self.parameters[i].values
            )

            param.update_and_step(cur_steps)

        self.iterations += 1


class Nadam(Adam):
    """Nadam: Nesterov Accelerated Adam.

    Combines Nesterov Momentum with RMSProp.
    """

    def step(self):
        m1 = self.first_momentum
        m2 = self.second_momentum
        it = self.iterations

        for i, param in enumerate(self.parameters):
            prev_fst_mma = self.fst_mom_mov_avg[i]
            prev_sec_mma = self.sec_mom_mov_avg[i]

            cur_fst_mma = m1 * prev_fst_mma + (1.0 - m1) * param.grads
            cur_sec_mma = m2 * prev_sec_mma + (1.0 - m2) * np.square(param.grads)

            if self.amsgrad:
                cur_sec_mma = np.maximum(prev_sec_mma, cur_sec_mma)

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            self.fst_mom_mov_avg[i] = cur_fst_mma
            self.sec_mom_mov_avg[i] = cur_sec_mma

            cur_fst_mma, prev_fst_mma = self._correct_bias(
                it, m1, cur_fst_mma, prev_fst_mma
            )
            cur_sec_mma = self._correct_bias(it, m2, cur_sec_mma)

            cur_lr = self.learning_rate / np.sqrt(cur_sec_mma + self.eps)
            decay_factor = self.weight_decay if self.params_should_decay[i] else 0.0

            cur_steps = cur_lr * (
                (1.0 + m1) * cur_fst_mma
                - m1 * prev_fst_mma
                + decay_factor * self.parameters[i].values
            )

            param.update_and_step(cur_steps)

        self.iterations += 1
