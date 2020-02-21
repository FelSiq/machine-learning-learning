"""Batch and Steady State evolutionary algorithms."""
import typing as t

import numpy as np

import evobasic


class EvoBatch(evobasic.EvoBasic):
    """Batch (Generational) evolutionary algorithm.

    +------------------------------+----------------+
    | Algorithm characteristic:    | Value:         |
    +------------------------------+----------------+
    | Overlapping population       | Is up to user  |
    +------------------------------+----------------+
    | Parent population size       | m > 0          |
    +------------------------------+----------------+
    | Offspring population size    | n > 0          |
    +------------------------------+----------------+
    | Parent selection scheme      | Is up to user  |
    +------------------------------+----------------+
    | Offspring selection scheme   | Is up to user  |
    +------------------------------+----------------+
    | Merge populations to select  | Is up to user  |
    +------------------------------+----------------+
    | Reproduction                 | Is up to user  |
    +------------------------------+----------------+
    | Mutation                     | Yes            |
    +------------------------------+----------------+
    | Crossover                    | Is up to user  |
    +------------------------------+----------------+
    """

    def __init__(self, *args, **kwargs):
        """Init a Batch (Generational) evolutionary model."""
        super().__init__(*args, **kwargs)

        self._alg_name = "Batch/Generational"

        self._offspring_pop = np.zeros(
            (self.pop_size_offspring, self.gene_num), dtype=np.float)
        self._offspring_timestamps = np.zeros(
            self.pop_size_offspring, dtype=np.uint)
        self._offspring_fitness = np.zeros(
            self.pop_size_offspring, dtype=np.float)

    def _reproduce(self) -> None:
        """Choose parents to reproduce."""
        num_parents_per_offspring = (1 + int(self.reproduction == "sexual"))
        num_offsprings = num_parents_per_offspring * self.pop_size_offspring

        id_parents = self._get_inst_ids(
            pop_size_target=num_offsprings,
            fitness_source=self.fitness,
            scheme=self.selection_parent,
            pick_best=True,
            args=self.selection_parent_args)

        if num_parents_per_offspring > 1:
            id_parents.reshape(-1, num_parents_per_offspring)

        for id_offspring, id_parents in enumerate(id_parents):
            offspring = self.reproduction_func(self.population[id_parents, :],
                                               **self.reproduction_func_args)

            offspring += [(p < self.mutation_prob[attr_ind]) *
                          self.mutation_delta_func[attr_ind](
                              **self.mutation_func_args[attr_ind])
                          for attr_ind, p in enumerate(
                              np.random.random(size=self.gene_num))]

            offspring = np.minimum(offspring, self.inst_range_high)
            offspring = np.maximum(offspring, self.inst_range_low)

            self._offspring_pop[id_offspring, :] = offspring
            self._offspring_timestamps[id_offspring] = self._time
            self._offspring_fitness[id_offspring] = self.fitness_func(
                offspring, **self.fitness_func_args)
            self._time += 1

    def _select(self) -> int:
        """Promote competition for a place in the next iteration."""
        killed_num = 0

        if self.overlapping_pops:
            if not self.merge_populations:
                # Scenario: promote duels between parents and offsprings
                id_targets = self._get_inst_ids(
                    pop_size_target=self.pop_size_offspring,
                    fitness_source=self.fitness,
                    scheme=self.selection_target,
                    pick_best=False,
                    args=self.selection_target_args)

                for id_offspring, id_kill in enumerate(id_targets):
                    if self.fitness[id_kill] < self._offspring_fitness[
                            id_offspring]:
                        self.population[id_kill, :] = self._offspring_pop[
                            id_offspring, :]
                        self.fitness[id_kill] = self._offspring_fitness[
                            id_offspring]
                        self.timestamps[id_kill] = self._offspring_timestamps[
                            id_offspring]
                        killed_num += 1

                return killed_num

            else:
                # Scenario: merge parents and offsprings, and then select
                target_pop = np.vstack((self.population, self._offspring_pop))
                target_fitness = np.concatenate((self.fitness,
                                                 self._offspring_fitness))
                target_timestamps = np.concatenate(
                    (self.timestamps, self._offspring_timestamps))
                calc_killed_num = lambda: np.sum(id_new_pop[:self.pop_size_parent] >= self.pop_size_parent)

        else:
            # Scenario: kill entire parent population, and select only offsprings
            target_pop = self._offspring_pop
            target_fitness = self._offspring_fitness
            target_timestamps = self._offspring_timestamps
            calc_killed_num = lambda: self.pop_size_parent

        id_new_pop = self._get_inst_ids(
            pop_size_target=self.pop_size_parent,
            fitness_source=target_fitness,
            scheme=self.selection_target,
            pick_best=True,
            args=self.selection_target_args)

        self.population = target_pop[id_new_pop, :]
        self.fitness = target_fitness[id_new_pop]
        self.timestamps = target_timestamps[id_new_pop]

        return calc_killed_num()

    def _gen_pop(self) -> t.Tuple[np.ndarray, int]:
        """Generate an entire batch of offsprings."""

        self._reproduce()
        killed_num = self._select()

        return self.population, killed_num


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
        pop_size_parent=1024,
        pop_size_offspring=512,
        gen_range_low=[0, 1],
        gen_range_high=[0, 1],
        gene_num=2,
        gen_num=64)

    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    _test_01()
    _test_02()
