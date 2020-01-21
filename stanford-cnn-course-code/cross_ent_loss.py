"""Implementation of the Cross Entropy loss.

Source: http://cs231n.github.io/linear-classify/
"""
import typing as t

import numpy as np
import scipy.special


def cross_ent_loss(X: np.ndarray,
                   y_inds: np.ndarray,
                   W: np.ndarray,
                   b: t.Optional[np.ndarray] = None,
                   lambda_: float = 0.0001) -> float:
    """Calculates the Cross Entropy loss.

    Source: http://cs231n.github.io/linear-classify/

    This loss is used in the `Softmax Classifier`, which is a
    generalization of the Logistic Regression classifier. This loss
    is actually just the cross entropy calculated on the softmax
    function, and assuming that the `True class distribuition` has
    a value 1 in the true class, and 0 in all other classes.

    Arguments
    ---------
    X : :obj:`np.ndarray`
        Array of instances. Each row is an instance, and each column is
        an instance attribute.

    y_indss : :obj:`np.ndarray`
        Indices of the correct class for every instance in ``X``.

    W : :obj:`np.ndarray`
        Weights of the previously fitted linear classifier.

    b : :obj:`np.ndarray`, optional
        Bias vector. If None, then ``X`` is expected to have the intercept
        column, and the bias vector is accordingly codded in within the
        ``W`` matrix.

    lambda_ : :obj:`float`, optional
        Scale factor for the regularization value. Zero value means no
        regularization. The regularization used is the Ridge regularization
        (sum of element-wise squared ``W``.)

    Returns
    -------
    float
        Total Cross Entropy loss of the given data.
    """
    _inst_inds = np.arange(y_inds.size)

    scores = np.dot(W, X.T)

    if b is not None:
        scores += b

    correct_class_score = scores[y_inds, _inst_inds]

    # Note: this is just the negative logarithm of the softmax function:
    # L = -\log\left(\frac{e^{f_{y_{i}}}}{\sum_{j}e^{f_{j}}}\right)
    #   = -f_{y_{i}} + \log\left(\sum_{j}e^{f_{j}}\right)
    loss = -correct_class_score + scipy.special.logsumexp(scores, axis=0)

    reg_factor = 0
    if not np.equal(0, lambda_):
        reg_factor = lambda_ * np.sum(np.square(W))

    return np.mean(loss) + reg_factor


def _test_01() -> None:
    np.random.seed(16)

    W = np.random.random((3, 5))
    X = np.random.randint(-5, 6, size=(10, 5))
    y = np.random.randint(3, size=10)

    loss_val = cross_ent_loss(X=X, y_inds=y, W=W)
    print("Loss:", loss_val)

    _aux_loss = np.mean(-np.log(
        scipy.special.softmax(np.dot(W, X.T), axis=0)[y, np.arange(y.size)]))
    _aux_reg = 0.0001 * np.sum(np.square(W))
    _total_loss = _aux_loss + _aux_reg

    assert np.isclose(loss_val, _total_loss)


def _test_02() -> None:
    W = np.array([
        [0.01, -0.05, 0.1, 0.05, 0],
        [0.7, 0.2, 0.05, 0.16, 0.2],
        [0.0, -0.45, -0.2, 0.03, -0.3],
    ])

    x = np.array([-15, 22, -44, 56, 1]).reshape(1, -1)

    print("Loss:", cross_ent_loss(X=x, y_inds=np.array([2]), W=W))


def _test_03() -> None:
    np.random.seed(16)

    NUM_INST = 10000
    NUM_CLASSES = 10
    IMG_DIM = np.prod((16, 16, 3))

    W = 0.002 * np.random.random(size=(NUM_CLASSES, IMG_DIM + 1)) - 0.001
    y = np.random.randint(NUM_CLASSES, size=NUM_INST)
    X = np.round(
        np.hstack((np.ones(
            (NUM_INST, IMG_DIM)) * y.reshape(-1, 1) + np.random.random(
                (NUM_INST, IMG_DIM)), np.ones((NUM_INST, 1)))))

    print("Loss:", cross_ent_loss(X=X, y_inds=y, W=W, lambda_=0))
    print("Expected:", np.log(NUM_CLASSES))


if __name__ == "__main__":
    _test_01()
    _test_02()
    _test_03()
