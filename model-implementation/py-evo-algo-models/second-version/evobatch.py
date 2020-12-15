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
    | Mutation scale adjust        | No             |
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
        offspring_parent_num = int(1 + int(self.reproduction == "sexual"))
        num_offsprings = offspring_parent_num * self.pop_size_offspring

        id_parents = self._get_inst_ids(
            pop_size_target=num_offsprings,
            fitness_source=self.fitness,
            scheme=self.selection_parent,
            pick_best=True,
            args=self.selection_parent_args)

        if offspring_parent_num > 1:
            id_parents = id_parents.reshape(-1, offspring_parent_num)

        offspring_pop_id = 0
        id_parent_group = 0

        while offspring_pop_id < self.pop_size_offspring:
            offspring = self.reproduction_func(
                self.population[id_parents[id_parent_group], :],
                **self.reproduction_func_args)

            if offspring.ndim != 1:
                num_offsprings = offspring.shape[0]
                if num_offsprings + offspring_pop_id >= self.pop_size_offspring:
                    offspring = offspring[:(
                        self.pop_size_offspring - offspring_pop_id), :]
                    num_offsprings = offspring.shape[0]
            else:
                num_offsprings = 1

            offspring += self._create_mutation(num_offsprings=num_offsprings)

            offspring = np.minimum(offspring, self.inst_range_high)
            offspring = np.maximum(offspring, self.inst_range_low)

            _aux_range = np.arange(offspring_pop_id,
                                   offspring_pop_id + num_offsprings)

            self._offspring_pop[_aux_range, :] = offspring
            self._offspring_timestamps[_aux_range] = self._time

            if num_offsprings == 1:
                self._offspring_fitness[_aux_range] = self.fitness_func(
                    offspring, **self.fitness_func_args)

            else:
                self._offspring_fitness[_aux_range] = np.apply_along_axis(
                    func1d=self.fitness_func,
                    arr=offspring,
                    axis=1,
                    **self.fitness_func_args)

            self._time += 1
            id_parent_group += 1
            offspring_pop_id += num_offsprings

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
        selection_parent_args={"size": 16},
        pop_size_parent=128,
        pop_size_offspring=256,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=48)
    model.run(verbose=True, plot=True)
    print(model)


def _test_02() -> None:
    import scipy.stats

    def fitness(inst, angle, r: float = 3.0, angle_adjust: float = 3e-4):
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

    model = EvoBatch(
        -8,
        8,
        fitness_func=fitness,
        fitness_func_args={"angle": [0.0]},
        mutation_delta_func=lambda: np.random.normal(0, 0.3),
        selection_parent="fitness-prop",
        selection_target="uniform",
        pop_size_parent=32,
        pop_size_offspring=32,
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
