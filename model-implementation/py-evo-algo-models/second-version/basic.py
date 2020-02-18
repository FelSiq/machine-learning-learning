"""Basic evolutionary model."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np

InstType = t.Union[float, np.ndarray, np.number, int]


class EvoBasic:
    """."""

    def __init__(self,
                 inst_range_low: InstType,
                 inst_range_high: InstType,
                 fitness_func: t.Callable[[InstType], t.Union[int, float]],
                 mutation_prob: t.Optional[InstType] = None,
                 mutation_delta_func: t.Optional[
                     t.Callable[[], t.Union[int, float]]] = None,
                 gen_num: int = 10,
                 inst_dim: int = 2,
                 size_pop: int = 128):
        """."""
        self.fitness_func = fitness_func

        if not np.isscalar(fitness_func(np.zeros(inst_dim, dtype=float))):
            raise TypeError("'fitness_func' must return a scalar.")

        if size_pop <= 0:
            raise ValueError(
                "'size_pop' must be positive (got {}.)".format(size_pop))

        if inst_dim <= 0:
            raise ValueError(
                "'inst_dim' must be positive (got {}.)".format(inst_dim))

        if gen_num <= 0:
            raise ValueError(
                "'gen_num' must be positive (got {}.)".format(gen_num))

        if (not np.isscalar(inst_range_low) and inst_range_low.size > 1
                and inst_range_low.size != inst_dim):
            raise ValueError("'inst_range_low' size ({}) does not match "
                             "with the instance dimension ({}). It must "
                             "be either 1 or assume the same size as the "
                             "'inst_dim' argument.".format(
                                 inst_range_low.size, inst_dim))

        if (not np.isscalar(inst_range_high) and inst_range_high.size > 1
                and inst_range_high.size != inst_dim):
            raise ValueError("'inst_range_high' size ({}) does not match "
                             "with the instance dimension ({}). It must "
                             "be either 1 or assume the same size as the "
                             "'inst_dim' argument.".format(
                                 inst_range_high.size, inst_dim))

        self.size_pop = size_pop
        self.inst_dim = inst_dim
        self.gen_num = gen_num

        self.inst_range_low = np.copy(inst_range_low)
        self.inst_range_high = np.copy(inst_range_high)

        self.population = np.array([])
        self.fitness = np.array([])
        self.timestamps = np.array([])

        self.best_inst_id = -1
        self.best_inst = np.array([])
        self.best_inst_fitness = -1.0

        self._online_plot = False

        if mutation_prob is None:
            mutation_prob = 1.0 / self.inst_dim

        if np.isscalar(mutation_prob):
            self.mutation_prob = np.full(
                shape=self.inst_dim, fill_value=mutation_prob)

        else:
            self.mutation_prob = np.copy(mutation_prob)

        if mutation_delta_func is not None:
            self.mutation_delta_func = mutation_delta_func

        else:
            self.mutation_delta_func = lambda: np.random.normal(loc=0, scale=1.0)

    def run(self,
            random_state: t.Optional[int] = None,
            verbose: bool = False,
            plot: bool = False,
            pause: float = 0.2,
            return_solution: bool = False) -> np.ndarray:
        """."""
        if random_state is not None:
            np.random.seed(random_state)

        self.population = np.random.uniform(
            low=self.inst_range_low,
            high=self.inst_range_high,
            size=(self.size_pop, self.inst_dim))

        self.timestamps = np.zeros(shape=self.size_pop, dtype=np.uint)

        self.fitness = np.array(
            [self.fitness_func(inst) for inst in self.population], dtype=float)

        if plot:
            self._config_plot(online=True)
            self.plot(pause=pause)

        for gen_ind in np.arange(self.gen_num):
            killed_num = 0

            for time, (id_parent, id_kill) in enumerate(
                    np.random.randint(self.size_pop, size=(self.size_pop, 2))):
                offspring = np.copy(self.population[id_parent])

                offspring += [(p < self.mutation_prob[attr_ind]) *
                              self.mutation_delta_func()
                              for attr_ind, p in enumerate(
                                  np.random.random(size=self.inst_dim))]

                offspring = np.minimum(offspring, self.inst_range_high)
                offspring = np.maximum(offspring, self.inst_range_low)

                offspring_fitness = self.fitness_func(offspring)

                if self.fitness[id_kill] < offspring_fitness:
                    self.population[id_kill, :] = offspring
                    self.fitness[id_kill] = offspring_fitness
                    self.timestamps[id_kill] = time
                    killed_num += 1

            if verbose:
                print("Generation {} finished. {} new instances.".format(
                    gen_ind, killed_num))

            if plot:
                self._plot_timestep(pause=pause)

        self.best_inst_id = np.argmax(self.fitness)
        self.best_inst = self.population[self.best_inst_id]
        self.best_inst_fitness = self.fitness[self.best_inst_id]
        self._online_plot = False

        if return_solution:
            return self.best_inst

        return self.population

    def _config_plot(self, online: bool = False) -> None:
        """."""
        if online:
            self._online_plot = True
            plt.ion()

        else:
            self._online_plot = False

        self._plt_fig = plt.figure()
        self._plt_ax = self._plt_fig.add_subplot(111)
        self._plt_con = None
        self._plt_sct = None

    def _plot_timestep(self, pause: float = 0.2):
        """."""
        if self._plt_sct:
            self._plt_sct.remove()

        if self.inst_dim == 2:
            self._plt_sct = self._plt_ax.scatter(
                self.population[:, 0], self.population[:, 1], color="blue")
        else:
            self._plt_sct = self._plt_ax.scatter(
                self.population, self.fitness, color="blue")

        plt.pause(pause)

    def plot(self, num_points: int = 64, pause: float = 0.0) -> None:
        """."""
        if not self.population.size:
            raise ValueError("No population to plot. Run 'run' method first.")

        if self.inst_dim > 2:
            raise ValueError(
                "Can't plot populations with more than 2 dimensions.")

        if not self._online_plot:
            self._config_plot(online=False)

        if self.inst_dim == 2:
            if self.inst_range_low.size == 2:
                vals_x = np.linspace(self.inst_range_low[0],
                                     self.inst_range_high[0], num_points)
                vals_y = np.linspace(self.inst_range_low[1],
                                     self.inst_range_high[1], num_points)
            else:
                vals_x = vals_y = np.linspace(self.inst_range_low,
                                              self.inst_range_high, num_points)

            X, Y = np.meshgrid(vals_x, vals_y)
            Z = np.zeros((num_points, num_points), dtype=float)

            for i in np.arange(num_points):
                for j in np.arange(num_points):
                    inst = np.array([X[i, j], Y[i, j]], dtype=float)
                    Z[i, j] = self.fitness_func(inst)

            self._plt_con = self._plt_ax.contour(
                X, Y, Z, levels=32, cmap="BuPu")

        else:
            vals = np.linspace(self.inst_range_low, self.inst_range_high,
                               num_points)

            self._plt_con = self._plt_ax.plot(
                vals, [self.fitness_func(val) for val in vals])

        plt.title("Population countour plot" + (" (best fit: {:.4f})".format(
            self.best_inst_fitness) if not self._online_plot else ""))
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")

        if self.inst_range_low.size > 1:
            plt.xlim(self.inst_range_low[0], self.inst_range_high[0])
            plt.ylim(self.inst_range_low[1], self.inst_range_high[1])

        else:
            plt.xlim(self.inst_range_low, self.inst_range_high)
            plt.ylim(self.inst_range_low, self.inst_range_high)

        try:
            self._plot_timestep(pause=pause)
            plt.show()

        except Exception:
            pass


def _test() -> None:
    model = EvoBasic(
        -5, 5, lambda inst: np.sum(inst * np.sin(inst)), inst_dim=1)
    model.run(verbose=True, plot=True)
    model.plot(pause=0)


if __name__ == "__main__":
    _test()
