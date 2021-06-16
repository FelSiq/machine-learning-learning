"""Implementation of the Loss functions.

Source: http://cs231n.github.io/linear-classify/
"""
import typing as t

import numpy as np
import scipy.special


def cross_ent_loss(
    X: np.ndarray,
    y_inds: np.ndarray,
    W: np.ndarray,
    b: t.Optional[np.ndarray] = None,
    scores: t.Optional[np.ndarray] = None,
) -> float:
    """Calculates the Cross Entropy loss.

    Source: http://cs231n.github.io/linear-classify/

    This loss is used in the `Softmax Classifier`, which is a
    generalization of the Logistic Regression classifier. This loss
    is actually just the cross entropy calculated on the softmax
    function, and assuming that the `True class distribuition` has
    a value 1 in the true class, and 0 in all other classes.

    Arguments
    ---------
    X : :obj:`np.ndarray`, shape: (num_inst, num_attr)
        Array of instances. Each row is an instance, and each column is
        an instance attribute.

    y_inds : :obj:`np.ndarray`, shape: (num_inst,)
        Indices of the correct class for every instance in ``X``.

    W : :obj:`np.ndarray`, shape: (num_classes, num_attr)
        Weights of the previously fitted linear classifier.

    b : :obj:`np.ndarray`, optional, shape: (num_inst,)
        Bias vector. If None, then ``X`` is expected to have the intercept
        column, and the bias vector is accordingly codded in within the
        ``W`` matrix.

    scores : :obj:`np.ndarray`, optional, shape: (num_classes, num_inst)
        Scores of each instance for each classe, calculated as the dot
        product between W and X.T (X transposed). If not given, then the
        scores will be calculated inside this function.

    Returns
    -------
    float
        Total Cross Entropy loss of the given data.
    """
    _inst_inds = np.arange(y_inds.size)

    if scores is None:
        scores = np.dot(W, X.T)

    if b is not None:
        scores += b

    correct_class_score = scores[y_inds, _inst_inds]

    # Note: this is just the negative logarithm of the softmax function:
    # L = -\log\left(\frac{e^{f_{y_{i}}}}{\sum_{j}e^{f_{j}}}\right)
    #   = -f_{y_{i}} + \log\left(\sum_{j}e^{f_{j}}\right)
    loss = -correct_class_score + scipy.special.logsumexp(scores, axis=0)

    return np.mean(loss)


def hinge_loss(
    X: np.ndarray,
    y_inds: np.ndarray,
    W: np.ndarray,
    b: t.Optional[np.ndarray] = None,
    scores: t.Optional[np.ndarray] = None,
    delta: float = 1.0,
) -> float:
    """Calculates the SVM loss for all instances in ``X``.

    The `SVM loss` is called `Multiclass Support Vector Machine loss`
    in the source document.

    Source: http://cs231n.github.io/linear-classify/

    Arguments
    ---------
    X : :obj:`np.ndarray`, shape: (num_inst, num_attr)
        Array of instances. Each row is an instance, and each column is
        an instance attribute.

    y_inds : :obj:`np.ndarray`, shape: (num_inst,)
        Indices of the correct class for every instance in ``X``.

    W : :obj:`np.ndarray`, shape: (num_classes, num_attr)
        Weights of the previously fitted linear classifier.

    b : :obj:`np.ndarray`, optional, shape: (num_inst,)
        Bias vector. If None, then ``X`` is expected to have the intercept
        column, and the bias vector is accordingly codded in within the
        ``W`` matrix.

    scores : :obj:`np.ndarray`, optional, shape: (num_classes, num_inst)
        Scores of each instance for each classe, calculated as the dot
        product between W and X.T (X transposed). If not given, then the
        scores will be calculated inside this function.

    delta : :obj:`float`, optional
        Size of the margin (distance) demanded between the true class
        score and the other scores.

    Returns
    -------
    float
        Total SVM loss in the given data.
    """
    _inst_inds = np.arange(y_inds.size)

    # Note: each row in 'X' is an instance
    # Note: each row in 'W' represents a distinct class
    if scores is None:
        scores = np.dot(W, X.T)

    if b is not None:
        scores += b

    correct_class_score = scores[y_inds, _inst_inds]

    # Note: maximum operates element-wise
    loss = np.maximum(0, scores - correct_class_score + delta)

    if not np.equal(delta, 0):
        loss[y_inds, _inst_inds] = 0

    el_wise_loss = np.sum(loss, axis=0)

    return np.mean(el_wise_loss)


def log_likelihood(
    X: np.ndarray,
    y_inds: np.ndarray,
    W: np.ndarray,
    b: t.Optional[np.ndarray] = None,
    scores: t.Optional[np.ndarray] = None,
) -> float:
    """Log-likelihood loss function."""
    # Note: each row in 'X' is an instance
    # Note: each row in 'W' represents a distinct class
    if scores is None:
        scores = np.dot(W, X.T)

    if b is not None:
        scores += b

    sig_out = 1.0 / (1.0 + np.exp(-scores))

    el_wise_loss = y_inds * np.log(sig_out) + (1 - y_inds) * np.log(1 - sig_out)

    return -np.mean(el_wise_loss)


def _test_hinge_01() -> None:
    np.random.seed(16)

    W = np.random.random((3, 5))
    X = np.random.randint(-5, 6, size=(10, 5))
    y = np.random.randint(3, size=10)

    print("Loss:", hinge_loss(X=X, y_inds=y, W=W))


def _test_hinge_02() -> None:
    W = np.array(
        [
            [0.01, -0.05, 0.1, 0.05, 0],
            [0.7, 0.2, 0.05, 0.16, 0.2],
            [0.0, -0.45, -0.2, 0.03, -0.3],
        ]
    )

    x = np.array([-15, 22, -44, 56, 1]).reshape(1, -1)

    print("Loss:", hinge_loss(X=x, y_inds=np.array([2]), W=W))


def _test_hinge_03() -> None:
    np.random.seed(16)

    NUM_INST = 100
    NUM_CLASSES = 10
    IMG_DIM = np.prod((16, 16, 3))

    W = np.full((NUM_CLASSES, IMG_DIM + 1), 1)
    y = np.random.randint(NUM_CLASSES, size=NUM_INST)
    X = np.round(
        np.hstack((np.random.random((NUM_INST, IMG_DIM)), np.ones((NUM_INST, 1))))
    )

    print("Loss:", hinge_loss(X=X, y_inds=y, W=W))
    print("Expected:", NUM_CLASSES - 1)


def _test_cross_ent_01() -> None:
    np.random.seed(16)

    W = np.random.random((3, 5))
    X = np.random.randint(-5, 6, size=(10, 5))
    y = np.random.randint(3, size=10)

    loss_val = cross_ent_loss(X=X, y_inds=y, W=W)
    print("Loss:", loss_val)

    _aux_loss = np.mean(
        -np.log(scipy.special.softmax(np.dot(W, X.T), axis=0)[y, np.arange(y.size)])
    )
    _aux_reg = 0.0001 * np.sum(np.square(W))
    _total_loss = _aux_loss + _aux_reg

    assert np.isclose(loss_val, _total_loss)


def _test_cross_ent_02() -> None:
    W = np.array(
        [
            [0.01, -0.05, 0.1, 0.05, 0],
            [0.7, 0.2, 0.05, 0.16, 0.2],
            [0.0, -0.45, -0.2, 0.03, -0.3],
        ]
    )

    x = np.array([-15, 22, -44, 56, 1]).reshape(1, -1)

    print("Loss:", cross_ent_loss(X=x, y_inds=np.array([2]), W=W))


def _test_cross_ent_03() -> None:
    np.random.seed(16)

    NUM_INST = 10000
    NUM_CLASSES = 10
    IMG_DIM = np.prod((16, 16, 3))

    W = 0.002 * np.random.random(size=(NUM_CLASSES, IMG_DIM + 1)) - 0.001
    y = np.random.randint(NUM_CLASSES, size=NUM_INST)
    X = np.round(
        np.hstack(
            (
                np.ones((NUM_INST, IMG_DIM)) * y.reshape(-1, 1)
                + np.random.random((NUM_INST, IMG_DIM)),
                np.ones((NUM_INST, 1)),
            )
        )
    )

    print("Loss:", cross_ent_loss(X=X, y_inds=y, W=W))
    print("Expected:", np.log(NUM_CLASSES))


if __name__ == "__main__":
    _test_cross_ent_01()
    _test_cross_ent_02()
    _test_cross_ent_03()
    _test_hinge_01()
    _test_hinge_02()
    _test_hinge_03()
