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


def _test() -> None:
    np.random.seed(16)

    W = np.random.random((3, 5))
    X = np.random.randint(-5, 6, size=(10, 5))
    y = np.random.randint(3, size=10)

    print("Loss:", cross_ent_loss(X=X, y_inds=y, W=W))


if __name__ == "__main__":
    _test()
