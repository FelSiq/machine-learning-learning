import typing as t
import warnings

import numpy as np
import sklearn.preprocessing
import sklearn.decomposition


class MLNetwork:
    """Implements a generic multilayer network."""

    @staticmethod
    def f_tanh(x: np.ndarray, alpha: float = 0.005) -> np.ndarray:
        """Hyperbolic tangent function recommended in reference paper.

        f(x) = 1.7159 * tanh(2/3 * x) + alpha * x

        Parameters
        ----------
        alpha : :obj:`float`
            Coefficient of the twisting term alpha * x to help avoiding
            flat spots.
        """
        return 1.7159 * np.tanh(0.6667 * x) + alpha * x

    @staticmethod
    def f_tanh_deriv(x: np.ndarray, alpha: float = 0.005) -> np.ndarray:
        """Derivative of the recommended activation function."""
        return 1.1440 * np.power(np.cosh(0.6667 * x), -2.0) + alpha

    @staticmethod
    def f_logistic(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def f_logistic_deriv(x: np.ndarray) -> np.ndarray:
        flog = MLNetwork.f_logistic(x)
        return flog * (1.0 - flog)

    @staticmethod
    def f_relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def f_relu_deriv(x: np.ndarray) -> np.ndarray:
        return np.heaviside(x, 0.0)

    @staticmethod
    def loss_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function."""
        return 0.5 * np.power(x - y, 2.0)

    @staticmethod
    def loss_function_deriv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function."""
        return y - x

    def __init__(self,
                 hidden_shape: t.Union[int, t.Tuple[int, ...]],
                 activation: str = "tanh"):
        """Init the network model."""
        self.hidden_shape = hidden_shape
        self.weights = None  # type: t.Optional[np.ndarray]
        self.learning_speed = None  # type: t.Optional[np.ndarray]
        self.layer_net = None  # type: t.Optional[np.ndarray]
        self.layer_output = None  # type: t.Optional[np.ndarray]

        self._ACTIVATION_FUNCS = {
            "tanh": (self.f_tanh, self.f_tanh_deriv),
            "logistic": (self.f_logistic, self.f_logistic_deriv),
            "relu": (self.f_relu, self.f_relu_deriv),
        }

        if activation not in self._ACTIVATION_FUNCS:
            warnings.warn(
                "Given 'activation' is unknown ({})"
                "".format(activation), UserWarning)

        self.act_fun, self.act_fun_deriv = self._ACTIVATION_FUNCS.get(
            activation, (self.f_tanh, self.f_tanh_deriv))

    def _init_weights(self, learning_speed: float) -> None:
        """Init weights at random."""
        shape = np.hstack((
            self._num_attr,
            self.hidden_shape,
            self._num_output,
        )).astype(np.int64)

        self.weights = [
            np.random.normal(
                loc=0.0,
                scale=np.power(shape[i], -0.5),
                size=(1 + shape[i]) * shape[i + 1],
            ).reshape((1 + shape[i], shape[i + 1]))
            for i in range(shape.size - 1)
        ]

        self.averaged_gradient = [
            np.repeat(
                learning_speed,
                (1 + shape[i]) * shape[i + 1]
            ).reshape((1 + shape[i], shape[i + 1]))
            for i in range(shape.size - 1)
        ]

        self.learning_speed = [
            np.repeat(
                np.sqrt(shape[i]) * learning_speed,
                (1 + shape[i]) * shape[i + 1]
            ).reshape((1 + shape[i], shape[i + 1]))
            for i in range(shape.size - 1)
        ]

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            learning_speed: float,
            normalize: bool = True,
            shuffle: bool = True,
            apply_pca: bool = False,
            random_seed: t.Optional[int] = None) -> "MLNetwork":
        """Fit data into model."""

        np.random.seed(random_seed)

        self.X = np.copy(X)
        self.y = np.copy(y)

        if normalize:
            self.X = self.X - self.X.mean(axis=0)

        if apply_pca:
            self.X = sklearn.decomposition.PCA(
                copy=False, random_state=random_seed).fit_transform(self.X)

        if normalize:
            self.X /= self.X.std(axis=0)

        if self.y.ndim == 1:
            self._num_classes = np.unique(self.y).size
            if self._num_classes > 2:
                self.y = sklearn.preprocessing.OneHotEncoder(
                    categories="auto", sparse=False).fit_transform(
                        self.y.reshape(-1, 1))
                self._num_output = self._num_classes

            else:
                self.y = self.y.reshape(-1, 1)
                self._num_output = 1

        else:
            self._num_classes = self.y.shape[1]
            self._num_output = self._num_classes

        self._num_inst = self.X.shape[0]
        self._num_attr = self.X.shape[1]

        if shuffle:
            indexes = np.arange(self._num_inst)
            np.random.shuffle(indexes)
            self.X = self.X[indexes, :]
            self.y = self.y[indexes, :]

        self._init_weights(learning_speed)

        return self

    def forward(self, instance: np.ndarray) -> np.ndarray:
        """Feed the network with the current instance."""
        net = instance.copy()

        num_layers = len(self.weights) + 1

        self.layer_net = num_layers * [None]
        self.layer_output = num_layers * [None]

        self.layer_net[0] = self.layer_output[0] = net

        for layer_index, layer_weights in enumerate(self.weights, 1):
            net = np.dot(np.hstack((net, 1.0)), layer_weights)

            # Save layer outputs and net for backpropagation
            self.layer_net[layer_index] = net
            net = self.layer_output[layer_index] = self.act_fun(net)

        return net

    def _backward(self, output: np.ndarray, cur_inst: np.ndarray,
                  cur_label: np.ndarray, momentum: float = 0.0,
                  ) -> t.Sequence[np.ndarray]:
        """Apply backpropagation and adjust network weights."""
        deltas = (self.act_fun_deriv(output) * self.loss_function_deriv(
            cur_label, output))

        adjust_weights = [
            np.multiply(
                np.dot(
                    np.vstack((self.layer_output[-2].reshape(-1, 1), 1.0)),
                    deltas.reshape(1, -1)),
                self.learning_speed[-1]
            )
        ]

        for i in np.arange(len(self.weights) - 1, 0, -1):
            cur_weights = self.weights[i - 1]

            prev_layer_neurons, cur_layer_neurons = cur_weights.shape

            cur_weight_adjust = np.zeros(cur_weights.shape)
            new_deltas = np.zeros(cur_layer_neurons)

            bias_out = np.hstack((self.layer_output[i - 1], 1.0))

            for j in np.arange(cur_layer_neurons):
                new_deltas[j] = (self.act_fun_deriv(self.layer_net[i][j]) *
                                 np.dot(deltas, self.weights[i][j, :]))

                for k in np.arange(prev_layer_neurons):
                    cur_weight_adjust[k, j] = (
                        self.learning_speed[i - 1][k, j] *
                        new_deltas[j] * bias_out[k])

            adjust_weights.insert(0, cur_weight_adjust)
            deltas = new_deltas

        return adjust_weights

    def _update_learning_speed(self,
                               alpha: float,
                               beta: float,
                               gamma: float,
                               adjust_weights: t.Sequence[np.ndarray]) -> None:
        """Update the learning speed of each weight."""
        for i in np.arange(len(self.averaged_gradient)):
            self.averaged_gradient[i] = (
                (1.0 - gamma)
                * self.averaged_gradient[i]
                + gamma * adjust_weights[i])

            self.learning_speed[i] = (
                self.learning_speed[i] + self.learning_speed[i] * alpha *
                (beta * np.linalg.norm(self.averaged_gradient[i], ord=2)
                 - self.learning_speed[i])
            )

    def train(self,
              batch_size: int = 1,
              epochs: int = 50,
              epsilon: float = 1.0e-8,
              update_learning_rate: bool = False,
              alpha: float = 0.0,
              beta: float = 0.0,
              gamma: float = 0.0,
              momentum: float = 0.0,
              it_to_print: t.Optional[int] = None) -> "MLNetwork":
        """Train the network weights with fitted data."""

        it_num = 0

        batch_mean_error = 1.0 + epsilon

        if it_to_print is None:
            it_to_print = epochs // 10

        while it_num < epochs and batch_mean_error > epsilon:
            it_num += 1

            batch_indexes = np.random.choice(
                a=self._num_inst, size=batch_size, replace=False)

            batch_mean_error = 0.0

            adjust_weights = [
                np.zeros(layer.shape) for layer in self.weights
            ]

            # Stochastic gradient descent
            for i in batch_indexes:
                cur_inst = self.X[i, :]
                cur_label = self.y[i, :]

                output = self.forward(cur_inst)

                new_adjust_weights = self._backward(
                    output=output,
                    cur_inst=cur_inst,
                    cur_label=cur_label,
                    momentum=momentum)

                for j in np.arange(len(self.weights)):
                    adjust_weights[j] += new_adjust_weights[j]

                batch_mean_error += self.loss_function(output, cur_label)

            for i in np.arange(len(self.weights)):
                self.weights[i] -= adjust_weights[i] + momentum * self.weights[i]

            if update_learning_rate:
                self._update_learning_speed(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    adjust_weights=adjust_weights)

            batch_mean_error = batch_mean_error.sum() / batch_size

            if it_to_print and (it_num % it_to_print == 0):
                print("Iteration id: {} - cur batch error: {}"
                      "".format(it_num, batch_mean_error))

        return self


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn import model_selection
    iris = datasets.load_iris()
    X = iris.data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = iris.target

    n_splits = 5

    sfk = model_selection.StratifiedKFold(n_splits=n_splits)

    acc = 0.0
    for train, test in sfk.split(X, y):
        labels = sklearn.preprocessing.OneHotEncoder(
            categories="auto", sparse=False).fit_transform(y.reshape(-1, 1))

        model = MLNetwork(5, activation="tanh")

        model.fit(
            X[train, :],
            labels[train, :],
            normalize=False,
            learning_speed=0.001)

        model.train(batch_size=int(0.1 * train.size), epochs=5000)

        cur_acc = 0.0
        for inst, lab in zip(X[test, :], labels[test, :]):
            prediction = np.round(model.forward(inst))
            cur_acc += np.allclose(prediction, lab)

        cur_acc /= test.size

        acc += cur_acc

    print("acc:", acc / n_splits)
    """
    func = "tanh"

    if func == "tanh":
        X = np.array([
            [ 1,  1],
            [ 1, -1],
            [-1, -1],
            [-1,  1],
        ])

        y = np.array([-1, 1, -1, 1])

    else:
        X = np.array([
            [1, 1],
            [1, 0],
            [0, 0],
            [0, 1],
        ])

        y = np.array([0, 1, 0, 1])

    model = MLNetwork(2, activation=func)
    model.fit(X, y, learning_speed=0.01, normalize=False)
    model.train(batch_size=4,
                epochs=500000,
                epsilon=1.0e-3,
                it_to_print=2000)

    if func == "tanh":
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    for inst in X:
        print(inst, model.forward(inst).round())
    """
