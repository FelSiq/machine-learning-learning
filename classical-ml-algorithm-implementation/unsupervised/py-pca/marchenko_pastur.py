"""Marchenko-pastur distribution median finding code.

Adapted (from Matlab to Python 3) from:
@misc{gavish2014optimal,
      title={The Optimal Hard Threshold for Singular Values is 4/sqrt(3)}, 
      author={Matan Gavish and David L. Donoho},
      year={2014},
      eprint={1305.5870},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}

Paper: https://arxiv.org/abs/1305.5870
Source code (MatLab code): https://purl.stanford.edu/vg705qn9070
"""
import numpy as np
import scipy.integrate


def median(beta):
    MarPas = lambda x: 1 - inc_mar_pas(x, beta, 0)
    lobnd = (1 - np.sqrt(beta)) ** 2
    hibnd = (1 + np.sqrt(beta)) ** 2
    change = True
    y = np.empty(5)
    while change and (hibnd - lobnd > 0.001):
        change = False
        x = np.linspace(lobnd, hibnd, 5)
        for i in range(1, len(x)):
            y[i] = MarPas(x[i])

        if any(y < 0.5):
            lobnd = max(x[y < 0.5])
            change = True

        if any(y > 0.5):
            hibnd = min(x[y > 0.5])
            change = True

    med = (hibnd + lobnd) * 0.5

    return med


def if_else(Q, point, counterPoint):
    y = point

    if np.any(~Q):
        if np.isscalar(counterPoint):
            counterPoint = np.full_like(Q, counterPoint)

        if np.isscalar(y):
            y = counterPoint[~Q]

        else:
            y[~Q] = counterPoint[~Q]

    return y


def inc_mar_pas(x0, beta, gamma):
    if beta > 1:
        raise RuntimeError("betaBeyond")

    topSpec = (1 + np.sqrt(beta)) ** 2
    botSpec = (1 - np.sqrt(beta)) ** 2
    MarPas = lambda x: if_else(
        (topSpec - x) * (x - botSpec) > 0,
        np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (2 * np.pi),
        0,
    )
    if np.isclose(gamma, 0):
        fun = lambda x: (x ** gamma * MarPas(x))
    else:
        fun = lambda x: MarPas(x)

    I, _ = scipy.integrate.quad(fun, x0, topSpec)
    return I
