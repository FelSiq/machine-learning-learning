"""Basic evolutionary model.

This model server as a common framework to all other evolutionary
algorithms in this package.
"""
import typing as t
import abc

import matplotlib.pyplot as plt
import numpy as np

InstType = t.Union[float, np.ndarray, np.number, int]
DeltaFuncType = t.Callable[[], t.Union[int, float]]
DeltaArgType = t.Dict[str, t.Any]


class EvoBasic:
    """Basic evolutionary model framework."""

    def __init__(self,
                 inst_range_low: InstType,
                 inst_range_high: InstType,
                 fitness_func: t.Callable[[InstType], t.Union[int, float]],
                 gene_num: int = 2,
                 pop_size_parent: int = 128,
                 pop_size_offspring: t.Optional[int] = None,
                 gen_num: int = 10,
                 mutation_prob: t.Optional[InstType] = None,
                 mutation_delta_func: t.Optional[
                     t.Union[DeltaFuncType, t.Sequence[DeltaFuncType]]] = None,
                 mutation_func_args: t.Optional[
                     t.Union[DeltaArgType, t.Sequence[DeltaArgType]]] = None,
                 gen_range_low: t.Optional[InstType] = None,
                 gen_range_high: t.Optional[InstType] = None):
        """Init the basic evolutionary algorithm framework.

        Arguments
        ---------
        inst_range_low : :obj:`np.ndarray` or :obj:`float`
            A scalar or numpy array for the lower bound for the genes
            (attributes) values. If a single number is given, the same
            lower bound will be used for all genes (attributes). If a
            numpy array is given, each position of the array must be
            the lower bound for the corresponding gene (i.e., the
            lower bound array must have dimension (gene_num,).)

        inst_range_high : :obj:`np.ndarray` or :obj:`float`
            The same as ``inst_range_low`` argument, but for the upper
            bound gene limit.

        fitness_func : callable
            Fitness function. Must receive a chromosome (instance) and
            return its respective fitness as a scalar value.

        gene_num : :obj:`int`, optional
            Number of genes for each chromosome, or the dimension of each
            instance/sample/individual.

        pop_size_parent : :obj:`int`, optional
            Size of the parent population. The greater is this number, the
            more parallelism (and less variance) is injected into the
            evolutionary algorithm search space.

        pop_size_offspring : :obj:`int`, optional
            Size of the offspring population, or the number of new samples
            (chromosomes) created in every evolutionary algorithm run (note
            that this is NOT the same as the number of offsprings in each
            generation.) The greater is this number, the more `delay` is
            injected into considering the information generated by the
            offsprings to generate new solutions.

        gen_num : :obj:`int`, optional
            Number of generations to run the evolutionary algorithm. One
            generation is defined as `number of new offsprings equal to
            the size of the parent population.`
            - If ``pop_size_parent`` > ``pop_size_offspring``, then the
            evolutionary algorithm is run more the once every generation.
            - If ``pop_size_parent`` == ``pop_size_offspring``, then the
            evolutionary algorithm is run exactly once every generation.
            - If ``pop_size_parent`` < ``pop_size_offspring``, then more
            than one generation is affected per evolutionary algorithm
            run.

        mutation_prob : :obj:`np.ndarray` or :obj:`float`, optional
            A scalar or numpy array giving the probability of mutation
            for each gene (attribute). If a single number is given,
            the same mutation probability is used to all genes. If an
            array is given, each position must be the mutation probability
            for the corresponding gene.

        mutation_delta_func : callable or a sequence of callables, optional
            Sampler of a distribution to generate the gene mutations.
            - If a single sampler is given, the same sampler will be used
            to generate mutation for all genes.
            - If a sequence is given, each position must generate the
            mutation for the corresponding gene.
            - If None given (None), then the mutation values will be drawn
            from a normal distribution of mean 0 and variance 1, for all
            genes.

        mutation_func_args : dict or a sequence of dicts, optional
            Arguments for the ``mutation_delta_func`` functions.
            - If a single dictionary is given, the same set of arguments
            will be given to every mutation function.
            - If a sequence of dictionary is given, then each position
            must kept the argumnets for each mutation function.
            - If None, no arguments will be passed to any mutation function.

            Note that, if this argument is not None, then the length of this
            argument must be either 1 (one dictionary), or match the length
            of the ``mutation_delta_func`` argument, if it is a sequence.

        gen_range_low : :obj:`np.ndarray` or :obj:`float`, optional
            The same as ``inst_range_low`` argument, but used only during the
            generation of the very first generation. If None, then the same
            value given by ``inst_range_low`` will be used.

        gen_range_high : :obj:`np.ndarray` or :obj:`float`, optional
            Same as ``gen_range_low``, but using ``inst_range_high`` as
            reference.
        """
        self.fitness_func = fitness_func

        if not np.isscalar(fitness_func(np.zeros(gene_num, dtype=float))):
            raise TypeError("'fitness_func' must return a scalar.")

        if pop_size_parent <= 0:
            raise ValueError("'pop_size_parent' must be positive (got {}.)".
                             format(pop_size_parent))

        if gene_num <= 0:
            raise ValueError(
                "'gene_num' must be positive (got {}.)".format(gene_num))

        if gen_num <= 0:
            raise ValueError(
                "'gen_num' must be positive (got {}.)".format(gen_num))

        if (not np.isscalar(inst_range_low) and inst_range_low.size > 1
                and inst_range_low.size != gene_num):
            raise ValueError("'inst_range_low' size ({}) does not match "
                             "with the instance dimension ({}). It must "
                             "be either 1 or assume the same size as the "
                             "'gene_num' argument.".format(
                                 inst_range_low.size, gene_num))

        if (not np.isscalar(inst_range_high) and inst_range_high.size > 1
                and inst_range_high.size != gene_num):
            raise ValueError("'inst_range_high' size ({}) does not match "
                             "with the instance dimension ({}). It must "
                             "be either 1 or assume the same size as the "
                             "'gene_num' argument.".format(
                                 inst_range_high.size, gene_num))

        if pop_size_offspring is None:
            pop_size_offspring = pop_size_parent

        if pop_size_offspring <= 0:
            raise ValueError("'pop_size_offspring' must be positive "
                             "(got {}.)".format(pop_size_offspring))

        self.pop_size_parent = pop_size_parent
        self.pop_size_offspring = pop_size_offspring
        self.gene_num = gene_num
        self.gen_num = gen_num

        if mutation_prob is None:
            mutation_prob = 1.0 / self.gene_num

        if np.any(not 0 <= mutation_prob <= 1):
            raise ValueError("'mutation_prob' must be in [0, 1] range "
                             "(got {}.)".format(mutation_prob))

        self.inst_range_low = np.copy(inst_range_low)
        self.inst_range_high = np.copy(inst_range_high)

        if gen_range_low is None:
            gen_range_low = self.inst_range_low

        if gen_range_high is None:
            gen_range_high = self.inst_range_high

        self.gen_range_low = gen_range_low
        self.gen_range_high = gen_range_high

        self.population = np.array([])
        self.fitness = np.array([])
        self.timestamps = np.array([])

        self.best_inst_id = -1
        self.best_inst = np.array([])
        self.best_inst_fitness = -1.0

        self._online_plot = False
        self._time = -1
        self._alg_name = None

        if np.isscalar(mutation_prob):
            self.mutation_prob = np.full(
                shape=self.gene_num, fill_value=mutation_prob)

        else:
            self.mutation_prob = np.copy(mutation_prob)

        if mutation_delta_func is None:
            mutation_delta_func = lambda: np.random.normal(loc=0, scale=1.0)

        if mutation_func_args is None:
            mutation_func_args = {}

        if callable(mutation_delta_func):
            self.mutation_delta_func = [
                mutation_delta_func for _ in np.arange(self.gene_num)
            ]

        else:
            self.mutation_delta_func = mutation_delta_func

        if isinstance(mutation_func_args, dict):
            self.mutation_func_args = [
                mutation_func_args for _ in np.arange(self.gene_num)
            ]

        else:
            self.mutation_func_args = mutation_func_args

        if len(self.mutation_func_args) != len(self.mutation_delta_func):
            raise ValueError("Length of 'mutation_func_args' ({}) and "
                             "'mutation_delta_func' ({}) differs.".format(
                                 len(self.mutation_func_args),
                                 len(self.mutation_delta_func)))

    def run(self,
            random_state: t.Optional[int] = None,
            verbose: bool = False,
            plot: bool = False,
            pause: float = 0.2,
            return_solution: bool = False) -> np.ndarray:
        """Run the selected evolutionary algorithm.

        Arguments
        ---------
        random_state : :obj:`int`, optional
            If given, set the random seed before any pseudo-random number
            generation, to keep the results reproducible.

        verbose : :obj:`bool`, optional
            If True, print information related to every epoch.

        plot : :obj:`bool`, optional
            If True, do online plotting during the algorithm execution.

        pause : :obj:`float`, optional
            Used only if ``plot`` is True. Number of seconds to wait
            before every epoch plot.

        return_solution : :obj:`np.ndarray`, optional
            If True, return the chromosome (instance) with the best fitness.
            If False, return the whole population after the algorithm
            execution.

        Returns
        -------
        np.ndarray
            If ``return_solution`` is True:
            - Return the chromosome (instance) with the best fitness.
            If ``return_solution`` is False:
            - Return the whole population after the algorithm execution.
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.population = np.random.uniform(
            low=self.gen_range_low,
            high=self.gen_range_high,
            size=(self.pop_size_parent, self.gene_num))

        self.timestamps = np.arange(self.pop_size_parent, dtype=np.uint)
        self._time = self.pop_size_parent

        self.fitness = np.array(
            [self.fitness_func(inst) for inst in self.population], dtype=float)

        if plot:
            self._config_plot(online=True)
            self.plot(pause=pause)

        gen_ind = 0
        current_it = 0
        killed_num = 0

        while gen_ind < self.gen_num:
            self.population, cur_killed_num = self._gen_pop()

            current_it += self.pop_size_offspring
            killed_num += cur_killed_num

            if current_it >= self.pop_size_parent:
                gen_ind += current_it // self.pop_size_parent
                current_it %= self.pop_size_parent

                if verbose:
                    print(
                        "Generation {} finished.".format(gen_ind),
                        "{} new instances.".format(killed_num)
                        if killed_num else "")

                killed_num = 0

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
        """Configure the class attributes related to plotting.

        Arguments
        ---------
        online : :obj:`bool`, optional
            If True, configure for online plotting (plot during the
            algorithm execution.)
        """
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
        """Plot the current population scatter plot.

        Arguments
        ---------
        pause : :obj:`float`, optional
            Number of seconds to wait before closing the current plot
            automatically.
        """
        if self._plt_sct:
            self._plt_sct.remove()

        if self.gene_num == 2:
            self._plt_sct = self._plt_ax.scatter(
                self.population[:, 0], self.population[:, 1], color="blue")
        else:
            self._plt_sct = self._plt_ax.scatter(
                self.population, self.fitness, color="blue")

        plt.pause(pause)

    def plot(self, num_points: int = 64, pause: float = 0.0) -> None:
        """Plot the fitness curve contour plot with the population scatter plot.

        Arguments
        ---------
        num_points : :obj:`int`, optional
            Number of points, for each dimension, for the contour plot.
            The higher is this value, the more precise will be the contour
            plot.

        pause : :obj:`float`, optional
            Number of seconds to wait before closing the current plot
            automatically.
        """
        if not self.population.size:
            raise ValueError("No population to plot. Run 'run' method first.")

        if self.gene_num > 2:
            raise ValueError(
                "Can't plot populations with more than 2 dimensions.")

        if not self._online_plot:
            self._config_plot(online=False)

        if self.gene_num == 2:
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

        plt.suptitle("Algorithm: {}".format(self._alg_name if self.
                                            _alg_name else "Unknown"))
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

    @abc.abstractmethod
    def _gen_pop(self) -> t.Tuple[np.ndarray, int]:
        """Generate a new population either in-place or not.

        Access the parent population using `self.population` from
        this superclass.

        Returns
        -------
        tuple of np.ndarray and int
            Tuple with the following values (in this order):
            - The numpy array with the new population
            - The number of killed chromosomes (instances) from the
                parent population.
        """
        return self.population, 0
