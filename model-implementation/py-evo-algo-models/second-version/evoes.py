"""Evolution Strategy evolutionary algorithm."""
import typing as t

import numpy as np

import evobatch


class EvoES(evobatch.EvoBatch):
    """Evolution Strategy evolutionary algorithm.

    +------------------------------+----------------+
    | Algorithm characteristic:    | Value:         |
    +------------------------------+----------------+
    | Overlapping population       | No             |
    +------------------------------+----------------+
    | Parent population size       | m > 0          |
    +------------------------------+----------------+
    | Offspring population size    | n >= m         |
    +------------------------------+----------------+
    | Parent selection scheme      | Uniform        |
    +------------------------------+----------------+
    | Offspring selection scheme   | Truncation     |
    +------------------------------+----------------+
    | Merge populations to select  | False          |
    +------------------------------+----------------+
    | Reproduction                 | Asexual        |
    +------------------------------+----------------+
    | Mutation                     | Yes            |
    +------------------------------+----------------+
    | Crossover                    | No             |
    +------------------------------+----------------+
    | Mutation scale adjust        | Yes            |
    +------------------------------+----------------+
    """

    def __init__(self,
                 mutation_scale_arg: str = "scale",
                 mutation_scale_adjust: float = 0.01,
                 *args,
                 **kwargs):
        """Init a Steady State evolutionary model.

        Arguments
        ---------
        mutation_scale_arg : :obj:`str`, optional
            Name of the argument fromm the ``mutation_func`` which controls
            the scale/step size of the mutations.
        """
        kwargs.setdefault("pop_size_parent", 1)
        kwargs.setdefault("pop_size_offspring", 10 * kwargs["pop_size_parent"])

        pop_size_offspring = kwargs.get("pop_size_offspring")

        if (pop_size_offspring is not None
                and pop_size_offspring < kwargs["pop_size_parent"]):
            raise ValueError("'pop_size_offspring' can't be smaller than "
                             "'pop_size_parent'.")

        super().__init__(
            overlapping_pops=False,
            merge_populations=False,
            selection_parent="uniform",
            selection_target="truncation",
            reproduction="asexual",
            reproduction_func=None,
            reproduction_func_args=None,
            *args,
            **kwargs)

        self._alg_name = "Evolution Strategy"
        self._mutation_scale_arg = mutation_scale_arg
        self._mutation_scale_adjust = mutation_scale_adjust

        for attr_args in self.mutation_func_args:
            attr_args.setdefault(self._mutation_scale_arg,
                                 np.random.normal(1.0, 0.05))

    def _adjust_mutation_scale(self,
                               acceptance_ratio: float,
                               target_acceptance_ratio: float = 0.2,
                               deviation_margin: float = 1e-2) -> None:
        """Adjust mutation step size based on current acceptance ratio.

        Arguments
        ---------
        acceptance_ratio : :obj:`float`
            Fraction of acceptance ratio from the last algorithm iteration.
            The acceptance ratio is the number of parents replaced by the
            offsprings by the parent population size.

        target_acceptance_ratio : :obj:`float`, optional
            The desired average acceptance ratio.

        deviation_margin : :obj:`float`, optional
            The accepted absolute maximum difference between ``acceptance_ratio``
            and ``target_acceptance_ratio``. The aboslute value of the difference
            must be higher than this argument in order to the scale be adjusted.
        """
        diff = acceptance_ratio - target_acceptance_ratio
        adjust_sign = np.sign(diff) if abs(diff) < deviation_margin else 0.0

        for attr_args in self.mutation_func_args:
            attr_args[self._mutation_scale_arg] *= (
                1.0 + adjust_sign * self._mutation_scale_adjust)

    def _gen_pop(self) -> t.Tuple[np.ndarray, int]:
        """Generate a new population and adjust the mutation scale."""
        self._reproduce()
        killed_num = self._select()
        self._adjust_mutation_scale(
            acceptance_ratio=killed_num / self.pop_size_parent)
        return self.population, killed_num


def _test_01() -> None:
    import numpy as np

    model = EvoES(
        inst_range_low=-8,
        inst_range_high=8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda scale: np.random.normal(0, scale),
        pop_size_parent=1,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=512)
    model.run(verbose=True, plot=True)
    print(model)


def _test_02() -> None:
    import scipy.stats

    def fitness(inst, angle, r: float = 3.0, angle_adjust: float = 1e-3):
        x1 = r * np.cos(angle[0])
        y1 = r * np.sin(angle[0])
        x2 = 1.5 * r * np.cos(np.pi - angle[0])
        y2 = 1.5 * r * np.sin(np.pi - angle[0])

        angle[0] = (angle[0] + angle_adjust) % 360

        a = scipy.stats.multivariate_normal.pdf(
            inst, mean=[x1, y1], cov=3 * np.eye(2))
        b = scipy.stats.multivariate_normal.pdf(
            inst, mean=[x2, y2], cov=1 * np.eye(2))
        c = scipy.stats.multivariate_normal.pdf(
            inst, mean=[x1, y1], cov=1 * np.eye(2))
        return 0.9 * a - 0.2 * c + 0.3 * b

    model = EvoES(
        inst_range_low=-8,
        inst_range_high=8,
        fitness_func=fitness,
        fitness_func_args={"angle": [0.0]},
        pop_size_parent=4,
        gene_num=2,
        gen_num=1024)

    model.run(
        verbose=True,
        plot_contour_points=8,
        time_invariant_fitness=False,
        plot=True,
        replot_fitness_contour=True)

    print(model)


if __name__ == "__main__":
    _test_01()
    _test_02()
