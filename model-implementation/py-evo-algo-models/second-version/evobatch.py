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
            pop_size_source=self.pop_size_parent,
            pop_size_target=self.pop_size_offspring,
            fitness_source=self.fitness,
            scheme=self.selection_parent,
            pick_best=True,
            args=self.selection_parent_args)

        if self.overlapping_pops:
            id_targets = self._get_inst_ids(
                pop_size_source=self.pop_size_parent,
                pop_size_target=self.pop_size_offspring,
                fitness_source=self.fitness,
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

        if self.overlapping_pops:
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

        else:
            offspring_fitness = np.apply_along_axis(
                func1d=self.fitness_func,
                arr=offspring_pop,
                axis=1,
                **self.fitness_func_args)

            id_new_pop = self._get_inst_ids(
                pop_size_source=self.pop_size_offspring,
                pop_size_target=self.pop_size_parent,
                fitness_source=offspring_fitness,
                scheme=self.selection_target,
                pick_best=True,
                args=self.selection_target_args)

            killed_num = self.pop_size_parent
            self.population = offspring_pop[id_new_pop, :]
            self.fitness = offspring_fitness[id_new_pop]
            self.timestamps = offspring_timestamps[id_new_pop]

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
        selection_parent="tournament",
        selection_target="uniform",
        selection_parent_args={"size": 128},
        pop_size_parent=512,
        pop_size_offspring=1024,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=48)
    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


def _test_02() -> None:
    def fitness(inst):
        x, y = inst
        if 0 < y <= 1 and 0 < x <= 4:
            return x + y
        if 3.75 <= x <= 4 and 0 <= y <= 4:
            return x + y

        x0, y0 = 3.75, 6
        r1 = 2
        r2 = 3
        x1 = np.sqrt(r1**2 - (y - y0)**2) + x0
        x2 = np.sqrt(r2**2 - (y - y0)**2) + x0
        if x1 <= x <= x2 and y0 - r1 <= y <= y0 + r2:
            return x + y

        if 5.0 <= x and 8 <= y <= 9:
            return 50 - 8 * x + y

        if 1.0 <= x < 5 and 8 <= y <= 8.5:
            return 30 - x

        return 0

    model = EvoBatch(
        np.array([0, 0]),
        np.array([7, 9]),
        fitness_func=fitness,
        selection_parent="uniform",
        selection_target="fitness-prop",
        selection_target_args={"size": 32},
        mutation_delta_func=lambda: np.random.normal(0, 0.5),
        pop_size_parent=1028,
        pop_size_offspring=512,
        gen_range_low=[0, 1],
        gen_range_high=[0, 1],
        gene_num=2,
        gen_num=64)

    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    # _test_01()
    _test_02()
