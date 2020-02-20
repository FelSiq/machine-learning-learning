"""Batch and Steady State evolutionary algorithms."""
import typing as t

import numpy as np

import evobasic


class EvoBatch(evobasic.EvoBasic):
    """Batch (Generational) evolutionary algorithm."""

    def __init__(self, *args, **kwargs):
        """Init a Batch (Generational) evolutionary model."""
        super().__init__(overlapping_pops=True, *args, **kwargs)
        self._alg_name = "Batch/Generational"

    def _gen_pop(self) -> t.Tuple[np.ndarray, int]:
        """Generate an entire batch of offsprings."""
        killed_num = 0

        offspring_pop = np.zeros((self.pop_size_offspring, self.gene_num),
                                 dtype=float)
        offspring_timestamps = np.zeros(self.pop_size_offspring, dtype=np.uint)

        id_parents = self._get_inst_ids(
            scheme=self.selection_parent,
            pick_best=True,
            args=self.selection_parent_args)
        id_targets = self._get_inst_ids(
            scheme=self.selection_target,
            pick_best=False,
            args=self.selection_target_args)

        for id_offspring, id_parent in enumerate(id_parents):
            offspring = np.copy(self.population[id_parent])

            offspring += [(p < self.mutation_prob[attr_ind]) *
                          self.mutation_delta_func[attr_ind](
                              **self.mutation_func_args[attr_ind])
                          for attr_ind, p in enumerate(
                              np.random.random(size=self.gene_num))]

            offspring = np.minimum(offspring, self.inst_range_high)
            offspring = np.maximum(offspring, self.inst_range_low)

            offspring_pop[id_offspring, :] = offspring
            offspring_timestamps[id_offspring] = self._time
            self._time += 1

        for id_offspring, id_kill in enumerate(id_targets):
            offspring = offspring_pop[id_offspring, :]
            offspring_ts = offspring_timestamps[id_offspring]

            offspring_fitness = self.fitness_func(offspring,
                                                  **self.fitness_func_args)

            if self.fitness[id_kill] < offspring_fitness:
                self.population[id_kill, :] = offspring
                self.fitness[id_kill] = offspring_fitness
                self.timestamps[id_kill] = offspring_ts
                killed_num += 1

        return self.population, killed_num


class EvoSteadyState(EvoBatch):
    """Steady State (batch size = 1) evolutionary algorithm."""

    def __init__(self, *args, **kwargs):
        """Init a Steady State (batch size = 1) evolutionary model."""
        super().__init__(pop_size_offspring=1, *args, **kwargs)
        self._alg_name = "Steady State/Incremental"


def _test_01() -> None:
    model = EvoBatch(
        -8,
        8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda: np.random.normal(0, 0.15),
        pop_size_parent=512,
        pop_size_offspring=1024,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=128)
    model.run(verbose=True, plot=True, pause=0.01)
    model.plot(pause=0)


"""
def _test_02() -> None:
    import scipy.stats
    def fitness(inst, mean):
        a = scipy.stats.multivariate_normal.pdf(inst, mean=mean, cov=1.0)
        b = scipy.stats.multivariate_normal.pdf(inst, mean=-mean, cov=1.0)
        return 0.55 * a + 0.45 * b

    model = EvoBatch(
        -8,
        8,
        fitness_func=
        mutation_delta_func=lambda: np.random.normal(0, 0.15),
        pop_size_parent=512,
        pop_size_offspring=1024,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=128)
    model.run(verbose=True, plot=True, pause=0.01)
    model.plot(pause=0)
"""

if __name__ == "__main__":
    _test_01()
    # _test_02()
