"""SVM loss implementation.

Source: http://cs231n.github.io/linear-classify/
"""
import typing as t

import numpy as np


def svm_loss(X: np.ndarray,
             y_inds: np.ndarray,
             W: np.ndarray,
             b: t.Optional[np.ndarray] = None,
             lambda_: float = 0.0001,
             delta: float = 1.0) -> float:
    """Calculates the SVM loss for all instances in ``X``.

    The `SVM loss` is called `Multiclass Support Vector Machine loss`
    in the source document.

    Source: http://cs231n.github.io/linear-classify/

    Arguments
    ---------
    X : :obj:`np.ndarray`
        Array of instances. Each row is an instance, and each column is
        an instance attribute.

    y_inds : :obj:`np.ndarray`
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
    scores = np.dot(W, X.T)

    if b is not None:
        scores += b

    correct_class_score = scores[y_inds, _inst_inds]

    # Note: maximum operates element-wise
    loss = np.maximum(0, scores - correct_class_score + delta)

    if not np.equal(delta, 0):
        loss[y_inds, _inst_inds] = 0

    el_wise_loss = np.sum(loss, axis=0)

    reg_factor = 0
    if not np.equal(lambda_, 0):
        reg_factor = lambda_ * np.sum(np.square(W))

    return np.mean(el_wise_loss) + reg_factor


def _test() -> None:
    np.random.seed(16)

    W = np.random.random((3, 5))
    X = np.random.randint(-5, 6, size=(10, 5))
    y = np.random.randint(3, size=10)

    print("Loss:", svm_loss(X=X, y_inds=y, W=W))


if __name__ == "__main__":
    _test()
