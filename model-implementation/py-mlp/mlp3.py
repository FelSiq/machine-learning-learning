import typing as t

import numpy as np
import sklearn.preprocessing
import sklearn.decomposition


class MLNetwork:
    """Implements a generic multilayer network."""

    @staticmethod
    def f_tanh(x: np.ndarray, alpha: float = 0.0) -> float:
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
    def f_tanh_deriv(x: np.ndarray, alpha: float = 0.0) -> float:
        """Derivative of the recommended activation function."""
        return 1.1440 * np.power(np.senh(0.6667 * x), 2.0) + alpha

    @staticmethod
    def loss_function(x: np.ndarray, y: np.ndarray):
        """Loss function."""
        return 0.5 * np.power(x - y, 2.0)

    def _init_weights(self, learning_speed: float):
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

    def __init__(self, hidden_shape: t.Union[int, t.Tuple[int, ...]]):
        """Init the network model."""
        self.hidden_shape = hidden_shape
        self.weights = None  # type: t.Optional[np.ndarray]
        self.learning_speed = None  # type: t.Optional[np.ndarray]
        self.layer_output = None  # type: t.Optional[np.ndarray]

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

    def forward(self, instance):
        """Feed the network with the current instance."""
        net = instance

        for layer_index, layer_weights in enumerate(self.weights):
            net = self.f_tanh(np.dot(np.hstack((net, 1.0)), layer_weights))
            # Save layer outputs for backpropagation
            self.layer_output[layer_index] = net

        return net

    def _backward(self, output: np.ndarray, cur_inst: np.ndarray) -> None:
        """Apply backpropagation and adjust network weights."""
        derivatives = None

        """To do."""

        for i in np.arange(self.weights.shape[0]):
            self.weights[i] -= np.multiply(self.learning_speed[i], derivatives)

    def train(self,
              batch_size: int = 1,
              epochs: int = 50,
              epsilon: float = 1.0e-8) -> "MLNetwork":
        """Train the network weights with fitted data."""

        it_num = 0

        prev_diff = 1.0 + epsilon
        batch_prev_error = batch_mean_error = 0.0

        while it_num < epochs and prev_diff < epsilon:
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
                cur_error = self.loss_function(output, cur_label)
                self._backward(cur_error, cur_inst)

                batch_mean_error += cur_error

            batch_mean_error /= batch_size
            prev_diff = abs(batch_mean_error - batch_prev_error)

        return self


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    model = MLNetwork(5)
    model.fit(iris.data, iris.target, learning_speed=0.01)

    print(model.forward(iris.data[1, :]))

    print(model.weights)
