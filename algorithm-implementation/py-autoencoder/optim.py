import numpy as np


class _BaseOptim:
    def __init__(self, learning_rate: float):
        assert float(learning_rate) > 0.0
        self.learning_rate = float(learning_rate)
        self.bias_correction = False

    def register_layer(self, *parameters):
        raise NotImplementedError

    def update(self, layer_id: int, *grads):
        raise NotImplementedError

    @staticmethod
    def _unpack(vals):
        return vals[0] if len(vals) == 1 else vals

    def _correct_bias(self, it: int, m: float, *updates):
        mom_it = m ** it

        if not self.bias_correction or mom_it < 1e-3:
            return self._unpack(updates)

        unbiased = []

        for update in updates:
            unbiased.append(update / (1.0 - mom_it))

        return self._unpack(unbiased)


class Momentum(_BaseOptim):
    """SGD with Momentum.

    Solves the difficulty of Vanilla SGD with surfaces that are too much
    steeper on only some directions than others. Since this algorithms adds
    a moving average to the model state position update, directions with
    high variability of direction cancel theirs own updates in average, while
    positions with low variability of direction builds up velocity and, hence,
    moves faster to the local minima.
    """

    def __init__(
        self,
        learning_rate: float,
        first_momentum: float,
        bias_correction: bool = True,
    ):
        super(Momentum, self).__init__(learning_rate)
        assert 1.0 > float(first_momentum) > 0.0
        self.first_momentum = float(first_momentum)
        self.bias_correction = bias_correction
        self.fst_mom_mov_avg = {}
        self._iterations = {}

    def register_layer(self, layer_id: int, *parameters):
        fst_mom_mov_avg = []

        for param in parameters:
            fst_mom_mov_avg.append(np.zeros_like(param))

        self.fst_mom_mov_avg[layer_id] = fst_mom_mov_avg
        self._iterations[layer_id] = 1

    def update(self, layer_id: int, *grads):
        m = self.first_momentum
        fst_mom_mov_avg = self.fst_mom_mov_avg[layer_id]
        it = self._iterations[layer_id]
        param_updates = []

        for i, grad in enumerate(grads):
            cur_vel = fst_mom_mov_avg[i]
            new_vel = m * cur_vel + (1.0 - m) * grad

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            fst_mom_mov_avg[i] = new_vel

            new_vel = self._correct_bias(it, m, new_vel)
            cur_updates = self.learning_rate * new_vel
            param_updates.append(cur_updates)

        self._iterations[layer_id] += 1

        return param_updates


class NesterovMomentum(Momentum):
    """Nesterov Accelerated Gradient (NAG).

    This optimizer modifies the vanilla (SGD+)Momentum algorithm with a
    'lookahead' of the gradients of the approximated next position
    considering the current state velocity and, hence, this optimizer
    builds up momentum in a informed way, and not completely blind like
    the vanilla Momentum algorithm.
    """

    def update(self, layer_id: int, *grads):
        m = self.first_momentum
        fst_mom_mov_avg = self.fst_mom_mov_avg[layer_id]
        it = self._iterations[layer_id]
        param_updates = []

        for i, grad in enumerate(grads):
            cur_vel = fst_mom_mov_avg[i]
            new_vel = m * cur_vel + (1 - m) * grad

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            fst_mom_mov_avg[i] = new_vel

            new_vel, cur_vel = self._correct_bias(it, m, new_vel, cur_vel)
            cur_updates = self.learning_rate * ((1.0 + m) * new_vel - m * cur_vel)
            param_updates.append(cur_updates)

        self._iterations[layer_id] += 1

        return param_updates


class Adagrad(_BaseOptim):
    """Adagrad learning rate.

    Not recommended since it is too aggressive while reducing the learning
    rates; the reduction is monotonic decreasing, and this may cause the
    learning to stop way too soon.
    """

    def __init__(self, learning_rate: float, eps: float = 1e-8):
        super(Adagrad, self).__init__(learning_rate)
        assert float(eps) > 0.0

        self.eps = float(eps)
        self.cum_sec_momentum = {}

    def register_layer(self, layer_id: int, *parameters):
        cum_sec_momentum = []

        for param in parameters:
            cum_sec_momentum.append(np.zeros_like(param))

        self.cum_sec_momentum[layer_id] = cum_sec_momentum

    def update(self, layer_id: int, *grads):
        cum_sec_momentum = self.cum_sec_momentum[layer_id]
        param_updates = []

        for i, grad in enumerate(grads):
            cur_cum_sec_mom = cum_sec_momentum[i]
            cur_cum_sec_mom += np.square(grad)

            cur_lr = self.learning_rate / np.sqrt(cur_cum_sec_mom + self.eps)
            cur_updates = cur_lr * grad
            param_updates.append(cur_updates)

        return param_updates


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
        learning_rate: float = 1.0,
        second_momentum: float = 0.9,
        eps: float = 1e-6,
    ):
        super(Adadelta, self).__init__(learning_rate=learning_rate)
        assert float(eps) > 0.0
        assert 1.0 > float(second_momentum) > 0.0

        self.eps = float(eps)
        self.second_momentum = float(second_momentum)
        self.mv_avg_second_mom_grads = {}
        self.mv_avg_second_mom_params = {}

    def register_layer(self, layer_id: int, *parameters):
        mv_avg_second_mom_grads = []
        mv_avg_second_mom_params = []

        for param in parameters:
            weights_grads, weights_params = np.zeros((2, *param.shape), dtype=float)
            mv_avg_second_mom_grads.append(weights_grads)
            mv_avg_second_mom_params.append(weights_params)

        self.mv_avg_second_mom_grads[layer_id] = mv_avg_second_mom_grads
        self.mv_avg_second_mom_params[layer_id] = mv_avg_second_mom_params

    def update(self, layer_id: int, *grads):
        mv_avg_second_mom_grads = self.mv_avg_second_mom_grads[layer_id]
        mv_avg_second_mom_params = self.mv_avg_second_mom_params[layer_id]
        param_updates = []
        m = self.second_momentum
        eps = self.eps

        for i, grad in enumerate(grads):
            cur_mov_avg_grads = mv_avg_second_mom_grads[i]
            cur_mov_avg_params = mv_avg_second_mom_params[i]

            cur_mov_avg_grads = m * cur_mov_avg_grads + (1 - m) * np.square(grad)

            # Note: weirdly enough, the 'eps' on the denominator prevents division
            # by zero, while the 'eps' on the numerator starts the recursive moving
            # average as the value of the first gradient instead of just zeros,
            # which is necessary to actually start the model training.
            cur_lr = np.sqrt((cur_mov_avg_params + eps) / (cur_mov_avg_grads + eps))
            cur_updates = cur_lr * grad

            cur_mov_avg_params = m * cur_mov_avg_params + (1 - m) * np.square(
                cur_updates
            )

            mv_avg_second_mom_grads[i] = cur_mov_avg_grads
            mv_avg_second_mom_params[i] = cur_mov_avg_params

            param_updates.append(cur_updates)

        return param_updates


class RMSProp(Adagrad):
    """RMSProp optimization algorithm.

    Like Adadelta, it was invented to mitigate the aggressiveness of the
    Adagrad algorithm. Although they were invented separately, RMSProp
    actually uses the same strategy from Adadelta: instead of keeping the
    sum of squares of all gradients, RMSProp (and Adadelta) uses a moving
    average to store the square of the gradients; in fact, Adadelta is
    just RMSProp with one step further of modification in the parameter
    update rule.
    """

    def __init__(
        self,
        learning_rate: float,
        second_momentum: float = 0.9,
        eps: float = 1e-8,
    ):
        assert 1.0 > float(second_momentum) > 0.0
        super(RMSProp, self).__init__(learning_rate=learning_rate, eps=eps)

        self.second_momentum = float(second_momentum)
        self.sec_mom_mov_avg = {}

    def register_layer(self, layer_id: int, *parameters):
        sec_mom_mov_avg = []

        for param in parameters:
            sec_mom_mov_avg.append(np.zeros_like(param))

        self.sec_mom_mov_avg[layer_id] = sec_mom_mov_avg

    def update(self, layer_id: int, *grads):
        sec_mom_mov_avg = self.sec_mom_mov_avg[layer_id]
        param_updates = []
        m = self.second_momentum

        for i, grad in enumerate(grads):
            cur_sec_mom_mov_avg = sec_mom_mov_avg[i]
            cur_sec_mom_mov_avg = m * cur_sec_mom_mov_avg + (1.0 - m) * np.square(grad)

            cur_lr = self.learning_rate / np.sqrt(cur_sec_mom_mov_avg + self.eps)
            cur_updates = cur_lr * grad
            param_updates.append(cur_updates)
            sec_mom_mov_avg[i] = cur_sec_mom_mov_avg

        return param_updates


class Adam(Momentum, RMSProp):
    """Adam optimization algorithm.

    Combines RMSProp with Momentum.
    """

    def __init__(
        self,
        learning_rate: float,
        first_momentum: float = 0.9,
        second_momentum: float = 0.999,
        eps: float = 1e-8,
    ):
        Momentum.__init__(
            self,
            learning_rate=learning_rate,
            first_momentum=first_momentum,
        )

        RMSProp.__init__(
            self,
            learning_rate=learning_rate,
            second_momentum=second_momentum,
            eps=eps,
        )

        self.bias_correction = True

    def register_layer(self, layer_id: int, *parameters):
        Momentum.register_layer(self, layer_id, *parameters)
        RMSProp.register_layer(self, layer_id, *parameters)

    def update(self, layer_id: int, *grads):
        fst_mom_mov_avg = self.fst_mom_mov_avg[layer_id]
        sec_mom_mov_avg = self.sec_mom_mov_avg[layer_id]
        param_updates = []
        m1 = self.first_momentum
        m2 = self.second_momentum
        it = self._iterations[layer_id]

        for i, grad in enumerate(grads):
            cur_fst_mma = fst_mom_mov_avg[i]
            cur_sec_mma = sec_mom_mov_avg[i]

            cur_fst_mma = m1 * cur_fst_mma + (1.0 - m1) * grad
            cur_sec_mma = m2 * cur_sec_mma + (1.0 - m2) * np.square(grad)

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            fst_mom_mov_avg[i] = cur_fst_mma
            sec_mom_mov_avg[i] = cur_sec_mma

            cur_fst_mma = self._correct_bias(it, m1, cur_fst_mma)
            cur_sec_mma = self._correct_bias(it, m2, cur_sec_mma)

            cur_lr = self.learning_rate / np.sqrt(cur_sec_mma + self.eps)
            cur_updates = cur_lr * cur_fst_mma
            param_updates.append(cur_updates)

        self._iterations[layer_id] += 1

        return param_updates


class Adamax(Adam):
    """Adam with second moment generalized to infinity norm.

    The main changes are listed below:
    - The second moment is not an moving average anymore; instead, it is
      an infinity norm between the scaled previous second moment and the
      absolute value of the current gradients.
    - Theres no need for bias correction onto the second moment anymore.
    """

    def update(self, layer_id: int, *grads):
        fst_mom_mov_avg = self.fst_mom_mov_avg[layer_id]
        sec_mom_mov_avg = self.sec_mom_mov_avg[layer_id]
        param_updates = []
        m1 = self.first_momentum
        m2 = self.second_momentum
        it = self._iterations[layer_id]

        for i, grad in enumerate(grads):
            cur_fst_mma = fst_mom_mov_avg[i]
            cur_sec_mma = sec_mom_mov_avg[i]

            cur_fst_mma = m1 * cur_fst_mma + (1.0 - m1) * grad
            cur_sec_mma = np.maximum(m2 * cur_sec_mma, np.abs(grad))

            # Note: do NOT store the unbiased version (calculated below), or else
            # everything will fall apart!
            fst_mom_mov_avg[i] = cur_fst_mma
            sec_mom_mov_avg[i] = cur_sec_mma

            cur_fst_mma = self._correct_bias(it, m1, cur_fst_mma)

            cur_lr = self.learning_rate / (cur_sec_mma + self.eps)
            cur_updates = cur_lr * cur_fst_mma
            param_updates.append(cur_updates)

        self._iterations[layer_id] += 1

        return param_updates
