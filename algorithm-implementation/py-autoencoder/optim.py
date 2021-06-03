import numpy as np


class _BaseOptim:
    def __init__(self, learning_rate: float):
        assert float(learning_rate) > 0.0
        self.learning_rate = float(learning_rate)

    def register_layer(self, *parameters):
        raise NotImplementedError

    def update(self, layer_id: int, *grads):
        raise NotImplementedError


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
        self, learning_rate: float, momentum: float, bias_correction: bool = True
    ):
        super(Momentum, self).__init__(learning_rate)
        assert 1.0 > float(momentum) > 0.0
        self.momentum = float(momentum)
        self.bias_correction = bias_correction
        self.velocity = {}
        self._iterations = {}

    def register_layer(self, layer_id: int, *parameters):
        vels = []

        for param in parameters:
            vels.append(np.zeros_like(param))

        self.velocity[layer_id] = vels
        self._iterations[layer_id] = 0

    def update(self, layer_id: int, *grads):
        m = self.momentum
        vels = self.velocity[layer_id]
        it = self._iterations[layer_id]
        ret = []

        for i, grad in enumerate(grads):
            cur_vel = vels[i]
            new_vel = m * cur_vel + (1.0 - m) * grad

            mom_it = m ** it

            if self.bias_correction and mom_it > 5e-2:
                new_vel /= 1.0 + mom_it

            vels[i] = new_vel

            param_updates = self.learning_rate * new_vel
            ret.append(param_updates)

        self._iterations[layer_id] += 1

        return ret


class NesterovMomentum(Momentum):
    """Nesterov Accelerated Gradient (NAG).

    This optimizer modifies the vanilla (SGD+)Momentum algorithm with a
    'lookahead' of the gradients of the approximated next position
    considering the current state velocity and, hence, this optimizer
    builds up momentum in a informed way, and not completely blind like
    the vanilla Momentum algorithm.
    """

    def update(self, layer_id: int, *grads):
        m = self.momentum
        vels = self.velocity[layer_id]
        it = self._iterations[layer_id]
        ret = []

        for i, grad in enumerate(grads):
            cur_vel = vels[i]
            new_vel = m * cur_vel + (1 - m) * grad

            mom_it = m ** it

            if self.bias_correction and mom_it > 5e-2:
                new_vel /= 1.0 + mom_it

            vels[i] = new_vel

            param_updates = self.learning_rate * (m * cur_vel + (1 + m) * new_vel)
            ret.append(param_updates)

        self._iterations[layer_id] += 1

        return ret


class Adagrad(_BaseOptim):
    """Adagrad learning rate.

    Not recommended since it is too agressive while reducing the learning
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
        ret = []

        for i, grad in enumerate(grads):
            cur_cum_sec_mom = cum_sec_momentum[i]
            cur_cum_sec_mom += np.square(grad)

            cur_lr = self.learning_rate / np.sqrt(cur_cum_sec_mom + self.eps)
            param_updates = cur_lr * grad
            ret.append(param_updates)

        return ret


class Adadelta(_BaseOptim):
    """Adadelta optimization algorithm.

    Solves the problem of the agressiveness of Adagrad, while preserving the
    adaptive learning rate per parameter.

    It is very similar to RMSProp optimizer, but was invented separately.

    The peculiar characteristic of Adadelta is that it does not depends on the
    learning rate hyperparameter.
    """

    def __init__(
        self, learning_rate: float = 1.0, momentum: float = 0.9, eps: float = 1e-6
    ):
        super(Adadelta, self).__init__(learning_rate=learning_rate)
        assert float(eps) > 0.0
        assert 1.0 > float(momentum) > 0.0

        self.eps = float(eps)
        self.momentum = float(momentum)
        self.mv_avg_second_mom_grads = {}
        self.mv_avg_second_mom_params = {}

    def register_layer(self, layer_id: int, *parameters):
        mv_avg_second_mom_grads = []
        mv_avg_second_mom_params = []

        for param in parameters:
            weights_grads, weights_params = np.zeros((2, *param.shape))
            mv_avg_second_mom_grads.append(weights_grads)
            mv_avg_second_mom_params.append(weights_params)

        self.mv_avg_second_mom_grads[layer_id] = mv_avg_second_mom_grads
        self.mv_avg_second_mom_params[layer_id] = mv_avg_second_mom_params

    def update(self, layer_id: int, *grads):
        mv_avg_second_mom_grads = self.mv_avg_second_mom_grads[layer_id]
        mv_avg_second_mom_params = self.mv_avg_second_mom_params[layer_id]
        ret = []
        m = self.momentum
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
            param_updates = cur_lr * grad

            cur_mov_avg_params = m * cur_mov_avg_params + (1 - m) * np.square(
                param_updates
            )

            mv_avg_second_mom_grads[i] = cur_mov_avg_grads
            mv_avg_second_mom_params[i] = cur_mov_avg_params

            ret.append(param_updates)

        return ret


class RMSProp(Adagrad):
    """."""

    def __init__(self, learning_rate: float, momentum: float = 0.9, eps: float = 1e-8):
        assert 1.0 > float(momentum) > 0.0
        super(RMSProp, self).__init__(learning_rate=learning_rate, eps=eps)

        self.momentum = float(momentum)
        self.sec_mom_mov_avg = {}

    def register_layer(self, layer_id: int, *parameters):
        sec_mom_mov_avg = []

        for param in parameters:
            sec_mom_mov_avg.append(np.zeros_like(param))

        self.sec_mom_mov_avg[layer_id] = sec_mom_mov_avg

    def update(self, layer_id: int, *grads):
        sec_mom_mov_avg = self.sec_mom_mov_avg[layer_id]
        ret = []
        m = self.momentum

        for i, grad in enumerate(grads):
            cur_sec_mom_mov_avg = sec_mom_mov_avg[i]
            cur_sec_mom_mov_avg = m * cur_sec_mom_mov_avg + (1.0 - m) * np.square(grad)

            cur_lr = self.learning_rate / np.sqrt(cur_sec_mom_mov_avg + self.eps)
            param_updates = cur_lr * grad
            ret.append(param_updates)
            sec_mom_mov_avg[i] = cur_sec_mom_mov_avg

        return ret
