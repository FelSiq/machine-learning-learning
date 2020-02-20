"""Batch and Steady State evolutionary algorithms."""
import typing as t

import numpy as np

import evobasic


class EvoBatch(evobasic.EvoBasic):
    """Batch (Generational) evolutionary algorithm."""

    def __init__(self, *args, **kwargs):
        """Init a Batch (Generational) evolutionary model."""
        super().__init__(*args, **kwargs)
        self._alg_name = "Batch/Gerational"

    def _gen_pop(self) -> t.Tuple[np.ndarray, int]:
        """Generate an entire batch of offsprings."""
        killed_num = 0

        offspring_pop = np.zeros((self.pop_size_offspring, self.gene_num),
                                 dtype=float)
        offspring_timestamps = np.zeros(self.pop_size_offspring, dtype=np.uint)

        id_parents, id_kills = np.random.randint(
            self.pop_size_parent, size=(2, self.pop_size_offspring))

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

        for id_offspring, id_kill in enumerate(id_kills):
            offspring = offspring_pop[id_offspring, :]
            offspring_ts = offspring_timestamps[id_offspring]

            offspring_fitness = self.fitness_func(offspring)

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


def _test() -> None:
    model = EvoSteadyState(
        -8,
        8,
        lambda inst: np.sum(inst[0] * np.sin(inst[1])),
        gene_num=2,
        gen_num=16)
    model.run(verbose=True, plot=True)
    model.plot(pause=0)


if __name__ == "__main__":
    _test()
