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
        ret = []

        vels = self.velocity[layer_id]
        it = self._iterations[layer_id]

        for i, grad in enumerate(grads):
            cur_vel = vels[i]
            grad = m * cur_vel + (1.0 - m) * grad

            if self.bias_correction:
                grad /= 1.0 + self.momentum ** it

            grad *= self.learning_rate
            ret.append(grad)

        self._iterations[layer_id] += 1

        return ret
