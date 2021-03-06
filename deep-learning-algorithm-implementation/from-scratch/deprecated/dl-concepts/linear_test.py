"""Test of a linear classifier in a image classification problem."""
import typing as t
import pickle

import matplotlib.pyplot as plt
import sgd_classifier
import sklearn.preprocessing
import numpy as np

NUM_BATCHES = 5
BATCH_SIZE = 10000
IMAGE_DIM = (3, 32, 32)


def unpickle(filepath: str) -> dict:
    """source: http://www.cs.toronto.edu/~kriz/cifar.html"""
    with open(filepath, "rb") as file_:
        data = pickle.load(file_, encoding="bytes")
    return data


def get_data(
    subpath: str, keep_classes: t.Optional[t.Sequence[int]] = None
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Get all CIFAR-10 data inside ``subpath`` folder.

    Data source: http://www.cs.toronto.edu/~kriz/cifar.html
    """
    if not subpath.endswith("/"):
        subpath += "/"

    X = np.zeros((NUM_BATCHES * BATCH_SIZE, np.prod(IMAGE_DIM)))
    y = np.zeros(NUM_BATCHES * BATCH_SIZE)

    for ind in np.arange(NUM_BATCHES):
        cur_batch = unpickle("{}data_batch_{}".format(subpath, ind + 1))

        cur_ind = ind * BATCH_SIZE

        X[cur_ind : (cur_ind + BATCH_SIZE), :] = cur_batch[b"data"]
        y[cur_ind : (cur_ind + BATCH_SIZE)] = cur_batch[b"labels"]

    if keep_classes is not None:
        insts = np.isin(y, keep_classes)
        X = X[insts, :]
        y = y[insts]

    return X, y.astype(int)


def plot(model: sgd_classifier.SGDClassifier) -> None:
    """Plot the representation of each class learnt by ``model``."""
    num_classes = model.weights.shape[0]
    scaled_coeffs = sklearn.preprocessing.minmax_scale(
        model.weights[:, :-1], (0, 1), axis=1
    )

    for cls_ind in np.arange(num_classes):
        if num_classes > 1:
            plt.subplot(2, num_classes // 2, 1 + cls_ind)

        plt.imshow(np.moveaxis(scaled_coeffs[cls_ind, :].reshape(IMAGE_DIM), 0, 2))

    plt.show()

    plt.title("Loss per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Average epoch loss")
    plt.plot(model.errors[0, :], linestyle="-", color="red", label="Train")
    plt.plot(model.errors[1, :], linestyle="-.", color="blue", label="Validation")
    plt.legend()

    plt.show()


def _test() -> None:
    """Plot the representation of each CIFAR-10 class by a linear model."""
    X, y = get_data(
        "/home/felipe/Documentos/cnn-stanford/data/cifar-10-batches-py",
        keep_classes=None,
    )

    np.random.seed(16)
    sample_inds = np.random.choice(y.size, y.size // 2, replace=False)
    X = X[sample_inds, :]
    y = y[sample_inds]

    # Scaling data into [-1, 1] is less important, but still helpful.
    scaler = sklearn.preprocessing.MinMaxScaler((-1, 1))
    X = scaler.fit_transform(X)

    model = sgd_classifier.SupportVectorClassifier()
    model.fit(
        X,
        y,
        patience=5,
        verbose=1,
        reg_rate=0.85,
        learning_rate=0.015,
        max_epochs=16,
        store_errors=True,
        random_state=16,
    )

    plot(model)


if __name__ == "__main__":
    _test()
