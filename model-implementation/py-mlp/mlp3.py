import typing as t
import warnings

import numpy as np
import sklearn.preprocessing
import sklearn.decomposition


class MLNetwork:
    """Implements a generic multilayer network."""

    @staticmethod
    def f_tanh(x: np.ndarray, alpha: float = 0.0) -> np.ndarray:
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
    def f_tanh_deriv(x: np.ndarray, alpha: float = 0.0) -> np.ndarray:
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
        }

        if activation not in self._ACTIVATION_FUNCS:
            warnings.warn("Given 'activation' is unknown ({})"
                          "".format(activation), UserWarning)

        self.act_fun, self.act_fun_deriv = self._ACTIVATION_FUNCS.get(
            activation, self.f_tanh)

    def _init_weights(self, learning_speed: float) -> None:
        """Init weights at random."""
        shape = np.hstack((
            self._num_attr,
            self.hidden_shape,
            self._num_classes,
        )).astype(np.int64)

        self.weights = np.array([
            np.random.normal(
                loc=0.0,
                scale=np.power(shape[i], -0.5),
                size=(1 + shape[i]) * shape[i + 1],
            ).reshape((1 + shape[i], shape[i + 1]))
            for i in range(shape.size - 1)
        ])

        self.learning_speed = np.array([
            np.repeat(learning_speed, n)
            for n in shape[1:]
        ])

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            learning_speed: float,
            normalize: bool = True,
            shuffle: bool = True,
            random_seed: t.Optional[int] = None) -> "MLNetwork":
        """Fit data into model."""

        np.random.seed(random_seed)

        self.X = np.copy(X)
        self.y = np.copy(y)

        if normalize:
            self.X = self.X - self.X.mean(axis=0)

        self.X = sklearn.decomposition.PCA(
            copy=False, random_state=random_seed).fit_transform(self.X)

        if normalize:
            self.X /= self.X.std(axis=0)

        self.y = sklearn.preprocessing.OneHotEncoder(
            categories="auto", sparse=False
        ).fit_transform(self.y.reshape(-1, 1))

        self._num_inst = self.X.shape[0]
        self._num_attr = self.X.shape[1]
        self._num_classes = self.y.shape[1]

        if shuffle:
            np.random.shuffle(self.X)

        self._init_weights(learning_speed)

        return self

    def forward(self, instance: np.ndarray) -> np.ndarray:
        """Feed the network with the current instance."""
        net = instance.copy()

        num_layers = self.weights.shape[0] + 1

        self.layer_net = num_layers * [None]
        self.layer_output = num_layers * [None]

        self.layer_net[0] = self.layer_output[0] = net

        for layer_index, layer_weights in enumerate(self.weights, 1):
            net = np.dot(np.hstack((net, 1.0)), layer_weights)

            # Save layer outputs and net for backpropagation
            self.layer_net[layer_index] = net
            self.layer_output[layer_index] = self.act_fun(net)

        return net

    def _backward(self,
                  output: np.ndarray,
                  cur_inst: np.ndarray,
                  cur_label: np.ndarray) -> None:
        """Apply backpropagation and adjust network weights."""
        deltas = (self.act_fun_deriv(output) *
                  self.loss_function_deriv(cur_label, output))

        # Adjust output layer later
        output_adjust = np.dot(
            np.vstack((self.layer_output[-2].reshape(-1, 1), 1.0)),
            (self.learning_speed[-1] * deltas).reshape(1, -1))

        for i in np.arange(self.weights.shape[0]-1, 0, -1):
            cur_weights = self.weights[i-1]

            weight_adjust = np.zeros(cur_weights.shape)
            cur_layer_neurons, prev_layer_neurons = cur_weights.shape

            new_deltas = np.zeros(cur_layer_neurons)

            for j in np.arange(cur_layer_neurons):
                cur_learning_speed = self.learning_speed[i-1][j]
                new_deltas[j] = (self.act_fun_deriv(self.layer_net[i][j]) *
                                 np.dot(deltas, self.weights[i][j]))

                for k in np.arange(prev_layer_neurons):
                    weight_adjust[k, j] = (
                        cur_learning_speed * new_deltas[j] *
                        np.hstack((self.layer_output[i-1], 1.0))[k])

            cur_weights -= weight_adjust
            deltas = new_deltas

        # Adjust the output layer
        self.weights[-1] -= output_adjust

    def train(self,
              batch_size: int = 1,
              epochs: int = 50,
              epsilon: float = 1.0e-8) -> "MLNetwork":
        """Train the network weights with fitted data."""

        it_num = 0

        prev_diff = 1.0 + epsilon
        batch_prev_error = batch_mean_error = 0.0

        while it_num < epochs and prev_diff > epsilon:
            it_num += 1

            batch_indexes = np.random.choice(
                a=self._num_inst,
                size=batch_size,
                replace=False)

            batch_prev_error = batch_mean_error
            batch_mean_error = 0.0

            # Stochastic gradient descent
            for i in batch_indexes:
                cur_inst = self.X[i, :]
                cur_label = self.y[i, :]

                output = self.forward(cur_inst)

                self._backward(
                    output=output,
                    cur_inst=cur_inst,
                    cur_label=cur_label)

                batch_mean_error += self.loss_function(output, cur_label)

            batch_mean_error = batch_mean_error.mean() / batch_size
            prev_diff = abs(batch_mean_error - batch_prev_error)

            print("Iteration id: {} - cur diff: {}".format(it_num, prev_diff))

        return self


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn import model_selection
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    sfk = model_selection.StratifiedKFold(n_splits=10)

    acc = 0.0
    for train, test in sfk.split(X, y):
        model = MLNetwork(5)
        model.fit(X[train, :], y[train], learning_speed=0.001)
        model.train(batch_size=train.size, epochs=200)

        labels = sklearn.preprocessing.OneHotEncoder(
            categories="auto", sparse=False
        ).fit_transform(y[test].reshape(-1, 1))

        cur_acc = 0.0
        for inst, lab in zip(X[test, :], labels):
            prediction = np.round(model.forward(inst))
            cur_acc += np.allclose(prediction, lab)

        cur_acc /= test.size

        acc += cur_acc

    print("acc:", acc / 10)
