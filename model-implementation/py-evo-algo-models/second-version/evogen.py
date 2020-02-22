"""Genetic Algorithm class."""
import typing as t

import numpy as np

import evobatch


class EvoGen(evobatch.EvoBatch):
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
    | Mutation scale adjust        | No             |
    +------------------------------+----------------+
    """

    def __init__(self,
                 crossover_points: t.Union[int, float] = 0.2,
                 produce_both_offsprings: bool = True,
                 *args,
                 **kwargs):
        """Init a genetic algorithm model.

        Arguments
        ---------
        crossover_points: :obj:`int` or :obj:`float`, optional
            - If integer, define the number of crossover points.
            - If float, this argument is the probability of the offspring
                getting the gene of the first parent. Must be a number in
                (0, 1) range.

        produce_both_offsprings : :obj:`bool`, optional
            If True, generate two offsprings for each crossover operation
            instead of a single one (if False.)
        """
        self._crossover_func = None
        self._crossover_args = {}

        if isinstance(crossover_points, (int, np.int)):
            if crossover_points <= 0:
                raise ValueError("'crossover_points' must be positive.")

            self._crossover_func = self.crossover_fixed
            self._crossover_args.setdefault("num_points", crossover_points)

        elif isinstance(crossover_points, (float, np.float)):
            if not 0.0 < crossover_points < 1.0:
                raise ValueError("'crossover_points' must be in (0, 1) range.")

            self._crossover_func = self.crossover_rand
            self._crossover_args.setdefault("gene_prob", crossover_points)

        else:
            raise TypeError(
                "'crossover_points' must be integer or float type.")

        self.produce_both_offsprings = produce_both_offsprings
        self._crossover_args.setdefault("return_both",
                                        self.produce_both_offsprings)
        super().__init__(
            overlapping_pops=False,
            merge_populations=False,
            pop_size_offspring=None,
            selection_parent="fitness-prop",
            selection_target="uniform",
            reproduction="sexual",
            reproduction_func=self._crossover_func,
            reproduction_func_args=self._crossover_args,
            *args,
            **kwargs)

        self._alg_name = "Genetic Algorithm"

        self._offspring_pop = np.zeros(
            (self.pop_size_offspring, self.gene_num), dtype=np.float)
        self._offspring_timestamps = np.zeros(
            self.pop_size_offspring, dtype=np.uint)
        self._offspring_fitness = np.zeros(
            self.pop_size_offspring, dtype=np.float)

    @staticmethod
    def _verify_parents(inst_a: np.ndarray,
                        inst_b: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Check both parent's dimensions, and return they."""
        if inst_b is None:
            if inst_a.ndim != 2:
                raise ValueError("'inst_b' is None and 'inst_a' has not two "
                                 "dimensions.")
            inst_b = inst_a[1, :]
            inst_a = inst_a[0, :]

        else:
            if inst_a.dtype != inst_b.dtype:
                raise TypeError(
                    "Data type of 'inst_a' and 'inst_b' must match.")

        if inst_a.size != inst_b.size:
            raise ValueError("Instances size does not match for crossover.")

        return inst_a, inst_b

    @classmethod
    def crossover_fixed(cls,
                        inst_a: np.ndarray,
                        inst_b: t.Optional[np.ndarray] = None,
                        num_points: int = 1,
                        return_both: bool = False) -> np.ndarray:
        """Crossover with a predetermined number of crosspoints.

        The crosspoints are chosen randomly with uniform probability.

        Arguments
        ---------
        inst_a : :obj:`np.ndarray`
            Array with both parents (if ``inst_b`` is None,) in each row, or
            just the first parent.

        inst_b : :obj:`np.ndarray`, optional
            Second parent. If not given, the second row of ``inst_a`` will
            be considered the second parent.

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
        inst_a, inst_b = cls._verify_parents(inst_a, inst_b)

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

    @classmethod
    def crossover_rand(cls,
                       inst_a: np.ndarray,
                       inst_b: t.Optional[np.ndarray] = None,
                       gene_prob: float = 0.2,
                       return_both: bool = False) -> np.ndarray:
        """Random crossover using coin flipping.

        With probability ``gene_prob``, each gene of the offspring will
        be from the first parent.

        Arguments
        ---------
        inst_a : :obj:`np.ndarray`
            Array with both parents (if ``inst_b`` is None,) in each row, or
            just the first parent.

        inst_b : :obj:`np.ndarray`, optional
            Second parent. If not given, the second row of ``inst_a`` will
            be considered the second parent.

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
        inst_a, inst_b = cls._verify_parents(inst_a, inst_b)

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


def _test() -> None:
    import numpy as np

    model = EvoGen(
        produce_both_offsprings=True,
        inst_range_low=-8,
        inst_range_high=8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda: np.random.normal(0, 0.1),
        pop_size_parent=16,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=96)
    model.run(verbose=True, plot=True)
    print(model)


if __name__ == "__main__":
    _test()
