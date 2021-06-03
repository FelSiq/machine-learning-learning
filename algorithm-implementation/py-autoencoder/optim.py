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

            grad = self.learning_rate * new_vel
            ret.append(grad)

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

            grad = self.learning_rate * (m * cur_vel + (1 + m) * new_vel)
            ret.append(grad)

        self._iterations[layer_id] += 1

        return ret
