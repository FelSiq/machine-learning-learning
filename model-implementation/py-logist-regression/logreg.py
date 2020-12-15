"""Simple Logist Regression implementation.

Helpful link:
https://beckernick.github.io/logistic-regression-from-scratch/
"""
import typing as t

import numpy as np


class LogReg:
    """Simple implementaiton of Logist Regression classification technique."""

    def __init__(self):
        self.X = None  # type: t.Optional[np.ndarray]
        self.y = None  # type: t.Optional[np.ndarray]
        self.weights = None  # type: t.Optional[np.ndarray]

    @staticmethod
    def sigmoid(val: t.Union[np.number, np.ndarray]):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-val))

    def log_likelihood(self, prediction: np.ndarray):
        """Log-likelihood of the model's prediction."""
        return sum(self.y * prediction - np.log(1.0 + np.exp(prediction)))

    def predict(self, query: np.ndarray, add_intercept: bool = True):
        """Perform a prediction using currently model weights."""
        if add_intercept:
            num_inst, _ = query.shape
            query = np.hstack((np.ones((num_inst, 1)), query))

        return LogReg.sigmoid(np.dot(query, self.weights))

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_it: int = 1000,
            learning_rate: float = 1.0e-4,
            add_intercept: bool = True,
            penalty: bool = True,
            C: float = 1.0) -> "LogReg":
        """Fit data into the model to adjust its weights."""
        self.X = np.array(X)
        self.y = np.array(y)

        num_inst, num_col = self.X.shape

        if add_intercept:
            self.X = np.hstack((np.ones((num_inst, 1)), self.X))
            num_col += 1

        self.weights = np.random.uniform(low=-0.5, high=0.5, size=num_col)

        C_inv = 1.0 / C

        iteration_id = 0
        while iteration_id < max_it:
            iteration_id += 1

            predictions = self.predict(self.X, add_intercept=False)

            error = self.y - predictions

            gradient = np.dot(self.X.T, error)

            if penalty:
                # L2 Regularization
                gradient += C_inv * self.weights

            self.weights += learning_rate * gradient

            if iteration_id % 1000 == 0:
                print("{0:<{fill}}: log-likelihood: "
                      "{1:<{fill}}".format(
                          iteration_id,
                          self.log_likelihood(predictions),
                          fill=12))

        return self


if __name__ == "__main__":
    # Test my model against Sklearn model using k-Fold CV.
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import KFold

    dataset = datasets.load_breast_cancer()

    data = dataset.data
    data = (data - data.mean()) / (data.std())
    target = dataset.target

    max_it = 500
    C = 1

    clf = LogisticRegression(fit_intercept=True, C=C, max_iter=max_it)
    n_splits = 20

    k_fold = KFold(n_splits=n_splits, shuffle=True)

    acc_my = np.zeros(n_splits)
    acc_sk = np.zeros(n_splits)

    for ids_id, ids_data in enumerate(k_fold.split(data)):
        ids_train, ids_test = ids_data

        x_train = data[ids_train, :]
        y_train = target[ids_train]

        x_test = data[ids_test, :]
        y_test = target[ids_test]

        model = LogReg().fit(
            x_train,
            y_train,
            max_it=max_it,
            learning_rate=0.001,
            add_intercept=True,
            C=C)

        clf.fit(x_train, y_train)

        pred_my = model.predict(x_test) >= 0.5
        pred_sk = clf.predict(x_test)

        acc_my[ids_id] = sum(y_test == pred_my) / y_test.size
        acc_sk[ids_id] = sum(y_test == pred_sk) / y_test.size

    print("Accuracy:\nsklearn = {0:0.4f} +/- {1:0.4f}\nmodel:"
          "{2:0.4f} +/- {3:0.4f}".format(acc_sk.mean(), acc_sk.std(),
                                         acc_my.mean(), acc_my.std()))
