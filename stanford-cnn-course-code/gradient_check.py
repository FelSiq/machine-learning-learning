"""Perform gradient checking to debug analytical gradients."""
import typing as t

import numpy as np

TypeVectorizedFunc = t.Callable[[t.Union[np.ndarray, float]], float]


def numerical_grad(func: TypeVectorizedFunc,
                   inst: t.Union[np.ndarray, float],
                   delta: float = 1.0e-5) -> np.float64:
    r"""Approximate the gradient of ``func`` evaluated on ``inst``.

    The strategy used is the centered formula:

        \frac{df(x)}{dx} = \frac{f(x + h) - f(x - h)}{2h}

    This centered formula has a order of magnitude higher O(h^2) of
    precision than the traditional finite difference approximation
    O(h).

        \frac{df(x)}{dx} = \frac{f(x + h) - f(x)}{h}
    """
    if not isinstance(inst, np.ndarray):
        inst = np.array([inst], dtype=np.float64)

    grad = np.zeros(inst.size, dtype=np.float64)
    inst_mod_a = np.copy(inst)
    inst_mod_b = np.copy(inst)

    _d_delta = np.float64(2.0 * delta)

    for dim_ind in np.arange(inst.size):
        inst_mod_a[dim_ind] += delta
        inst_mod_b[dim_ind] -= delta
        grad[dim_ind] = (func(inst_mod_a) - func(inst_mod_b)) / _d_delta
        inst_mod_a[dim_ind] -= delta
        inst_mod_b[dim_ind] += delta

    if inst.size == 1:
        return grad[0]

    return grad


def _gen_random_data(x_limits: t.Sequence[t.Tuple[int, int]],
                     num_it: int = 20000,
                     random_state: t.Optional[int] = None) -> np.ndarray:
    """Generate random data for gradient checking."""
    if random_state is not None:
        np.random.seed(random_state)

    if len(x_limits) == 1:
        lim_low, lim_upper = x_limits[0][0], x_limits[0][1]
        x_rand = np.random.uniform(lim_low, lim_upper, size=num_it)

    else:
        x_rand = np.hstack([
            np.random.uniform(lim_low, lim_upper, size=(num_it, 1))
            for lim_low, lim_upper in x_limits
        ])

    return x_rand.astype(np.float64)


def gradient_check(func: TypeVectorizedFunc,
                   analytic_grad: TypeVectorizedFunc,
                   x_limits: t.Sequence[t.Tuple[int, int]],
                   delta: float = 1.0e-5,
                   num_it: int = 20000,
                   verbose: int = 0,
                   random_state: t.Optional[int] = None) -> float:
    """Check if the analytical gradient matches with the numerical.

    This is a debugging tool to check if the analitical gradient
    was calculated correctly.

    The strategy adopted is a monte-carlo test, where a huge number
    of random tests are tried.

    Arguments
    ---------
    func : :obj:`callable`
        Function corresponding to the gradient to be tested. Must
        receive an p-dimensional value as the first argument, and
        return a corresponding scalar as the function image.

    analytic_grad : :obj:`callable`
        Analytical gradient of ``func``. Must receive a p-dimensional
        value `x` as first argument, and return the corresponding
        gradient vector of `x`. For instance, suppose that

            func = lambda x: 2 * x[0]**3 - 5 * x[1]**2 + 7

        Therefore, the ``analytic_grad`` must return an two-dimensional
        vector with the following partial derivatives:

            analytic_grad = lambda x: [6 * x[0]**2, -10 * x[1]]

    x_limits : :obj:`sequence` of :obj:`tuple` (int, int)
        Sequence of numerical limits for every dimension of `func`
        to be tested. Each entry of this sequence corresponds to
        one dimension, and every entry is a pair of values in the
        form `(lower_limit, upper_limit).`

    delta : :obj:`float`, optional
        A tiny value to calculate the numerical gradient of the form
        (to a 1-dimensional instance):

            num_grad(x) = (func(x + delta) - func(x)) / delta

        For p-dimensional instances, p > 1, the formula above is
        repeated for every dimension separately.

    num_it : :obj:`int`, optional
        Number of random tests to be drawn.

    verbose : :obj:`int`, optional
        Verbosity level of the function.

    random_state : :obj:`int`, optional
        If given, set the random seed before any pseudo-random number
        generation. Keeps the results reproducible.

    Returns
    -------
    float
        Average max norm between analytical and numerical gradient
        strategies.
    """
    x_rand = _gen_random_data(
        x_limits=x_limits, num_it=num_it, random_state=random_state)

    rel_total_err = np.float64(0.0)

    for cur_it, inst in enumerate(x_rand):
        val_num_grad = numerical_grad(func=func, inst=inst, delta=delta)
        val_ana_grad = analytic_grad(inst)

        abs_diff = np.abs(
            val_num_grad.astype(np.float64) - val_ana_grad.astype(np.float64))

        max_el_wise = np.maximum(np.abs(val_num_grad), np.abs(val_ana_grad))

        _non_zero_inds = np.logical_and(abs_diff > 1e-9, max_el_wise > 1e-8)

        abs_diff[_non_zero_inds] /= np.maximum(
            np.abs(val_num_grad[_non_zero_inds]),
            np.abs(val_ana_grad[_non_zero_inds]))

        rel_total_err += abs_diff

        if verbose >= 2:
            print("Current iteration: {} - Current total relative error "
                  "maximum norm: {}".format(cur_it, np.max(rel_total_err)))

    avg_max_norm_err = np.max(rel_total_err) / num_it

    if verbose:
        if avg_max_norm_err > 1e-2:
            _err_msg = "The analytical gradient is probably wrong"

        elif 1e-2 >= avg_max_norm_err > 1e-4:
            _err_msg = "Something may be wrong with the analytical gradient"

        elif 1e-4 >= avg_max_norm_err > 1e-7:
            _err_msg = ("If the function is simple, then error is too "
                        "high. Otherwise, it is ok")

        else:
            _err_msg = "Gradient is ok"

        print("Average relative error maximum norm: {:.10f}".format(
            avg_max_norm_err))

        print("Conclusion: {}.".format(_err_msg))

    return avg_max_norm_err


def _test_01() -> None:
    error = gradient_check(
        func=lambda x: 5 * x**3 - 3 * x + 1,
        analytic_grad=lambda x: 15 * x**2 - 3,
        x_limits=[(-5, 5)],
        random_state=16)

    print(error)


def _test_02() -> None:
    error = gradient_check(
        func=lambda x: 5 * x[0]**3 - 3 * x[1]**2 + 1,  # type: ignore
        analytic_grad=
        lambda x: np.array([15 * x[0]**2, -6 * x[1]]),  # type: ignore
        x_limits=[(-5, 5), (-10, 10)],
        random_state=32)

    print(error)


if __name__ == "__main__":
    _test_01()
    _test_02()
