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
        self.mv_avg_second_mom = {}

    def register_layer(self, layer_id: int, *parameters):
        mv_avg_second_mom = []

        for param in parameters:
            mv_avg_second_mom.append(np.zeros_like(param))

        self.mv_avg_second_mom[layer_id] = mv_avg_second_mom

    def update(self, layer_id: int, *grads):
        mv_avg_second_mom = self.mv_avg_second_mom[layer_id]
        ret = []

        for i, grad in enumerate(grads):
            cur_mov_avg = mv_avg_second_mom[i]
            cur_mov_avg += np.square(grad)

            cur_lr = self.learning_rate / np.sqrt(cur_mov_avg + self.eps)
            param_updates = cur_lr * grad
            ret.append(param_updates)

        return ret


class Adadelta(_BaseOptim):
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
