"""Genetic Algorithm class."""
import evobasic


class EvoGen(evobasic.EvoBasic):
    """Genetic Algorithm evolutionary algorithm.

    +------------------------------+----------------+
    | Algorithm characteristic:    | Value:         |
    +------------------------------+----------------+
    | Overlapping population       | False          |
    +------------------------------+----------------+
    | Parent population size       | m > 0          |
    +------------------------------+----------------+
    | Offspring population size    | m              |
    +------------------------------+----------------+
    | Parent selection scheme      | Fitness-prop   |
    +------------------------------+----------------+
    | Offspring selection scheme   | Uniform        |
    +------------------------------+----------------+
    | Merge populations to select  | False          |
    +------------------------------+----------------+
    | Reproduction                 | Sexual         |
    +------------------------------+----------------+
    | Mutation                     | Yes            |
    +------------------------------+----------------+
    | Crossover                    | Yes            |
    +------------------------------+----------------+
    """

    def __init__(self, crossover_points: t.Union[int, float], *args, **kwargs):
        """Init a genetic algorithm model.

        Arguments
        ---------
        crossover_points: :obj:`int` or :obj:`float`
            - If integer, define the number of crossover points.
            - If float, this argument is the probability of the offspring
                getting the gene of the first parent. Must be a number in
                (0, 1) range.
        """
        self._crossover_func = None
        self.crossover_param = crossover_points

        if isinstance(crossover_points, (int, np.int)):
            if crossover_points <= 0:
                raise ValueError("'crossover_points' must be positive.")

            self._crossover_func = self.crossover_fixed

        elif isinstance(crossover_points, (float, np.float)):
            if not 0 < crossover_points < 1:
                raise ValueError("'crossover_points' must be in (0, 1) range.")

            self._crossover_func = self.crossover_rand

        else:
            raise TypeError(
                "'crossover_points' must be integer or float type.")

        super().__init__(
            overlapping_pops=False,
            merge_populations=False,
            pop_size_offspring=None,
            selection_parent="fitness-prop",
            selection_target="uniform",
            *args,
            **kwargs)

        self._alg_name = "Genetic Algorithm"

    @staticmethod
    def crossover_fixed(inst_a: np.ndarray,
                        inst_b: np.ndarray,
                        num_points: int = 1,
                        return_both: bool = False) -> np.ndarray:
        """Crossover with a predetermined number of crosspoints.

        The crosspoints are chosen randomly with uniform probability.

        Arguments
        ---------
        num_points : :obj:`int`, optional
            Number of crosspoints.

        return_both : :obj:`bool`, optional
            If True, return the children and its `conjugate` (the children
            produce by the genes not chosen.)

        Returns
        -------
        np.ndarray
            If ``return_both`` is True, return a two dimensional array where
            each row is a produced children. Otherwise, return only the first
            children (the `non-conjugate` one.)
        """
        if inst_a.size != inst_b.size:
            raise ValueError("Instances size does not match for crossover.")

        if inst_a.dtype != inst_b.dtype:
            raise TypeError("Data type of 'inst_a' and 'inst_b' must match.")

        cuts = np.zeros(inst_a.size, dtype=np.uint)
        cuts[np.random.choice(inst_a.size - 1, size=num_points,
                              replace=False)] = 1

        offspring = np.zeros((2, inst_a.size), dtype=inst_a.dtype)
        insts = (inst_a, inst_b)
        inst_ind = 0

        for cur_ind, cut in enumerate(cuts):
            offspring[0, cur_ind] = insts[inst_ind][cur_ind]
            offspring[1, cur_ind] = insts[1 - inst_ind][cur_ind]

            if cut: inst_ind = 1 - inst_ind

        if return_both:
            return offspring

        return offspring[0, :]

    @staticmethod
    def crossover_rand(inst_a: np.ndarray,
                       inst_b: np.ndarray,
                       gene_prob: float = 0.2,
                       return_both: bool = False) -> np.ndarray:
        """Random crossover using coin flipping.

        With probability ``gene_prob``, each gene of the offspring will
        be from the first parent.

        Arguments
        ---------
        gene_prob : :obj:`float`, optional
            Probability of the offspring copy each gene from the first parent
            (``inst_a``). Must be in [0, 1] range.

        return_both : :obj:`bool`, optional
            If True, return the children and its `conjugate` (the children
            produce by the genes not chosen.)

        Returns
        -------
        np.ndarray
            If ``return_both`` is True, return a two dimensional array where
            each row is a produced children. Otherwise, return only the first
            children (the `non-conjugate` one.)
        """
        if inst_a.size != inst_b.size:
            raise ValueError("Size of instances does not match for crossover.")

        if inst_a.dtype != inst_b.dtype:
            raise TypeError("Data type of both instances must match.")

        cuts = np.random.choice(
            2, size=inst_a.size, replace=True,
            p=[gene_prob, 1.0 - gene_prob]).astype(np.uint)

        offspring = np.zeros((2, inst_a.size), dtype=inst_a.dtype)

        inds_a = cuts == 0
        inds_b = ~inds_a

        offspring[:, inds_a] = inst_a[inds_a], inst_b[inds_a]
        offspring[:, inds_b] = inst_b[inds_b], inst_a[inds_b]

        if return_both:
            return offspring

        return offspring[0, :]

    def _reproduce(self) -> None:
        """."""

    def _gen_pop(self) -> t.Union[np.ndarray, int]:
        """."""
        self._reproduce()
        killed_num = self._select()
        return self.population, killed_num


def _test() -> None:
    import numpy as np

    model = EvoGen(
        -8,
        8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda: np.random.normal(0, 0.1),
        pop_size_parent=16,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=96)
    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    _test()
