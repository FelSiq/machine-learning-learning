"""Evolution Strategy evolutionary algorithm."""
import evobatch

class EvoSteadyState(evobatch.EvoBatch):
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
    """

    def __init__(self, *args, **kwargs):
        """Init a Steady State evolutionary model."""
        kwargs.setdefault("pop_size_parent", 1)
        kwargs.setdefault("pop_size_offspring", 10 * kwargs["pop_size_parent"])

        pop_size_offspring = kwargs.get("pop_size_offspring")

        if (pop_size_offspring is not None and
                pop_size_offspring < kwargs["pop_size_parent"]):
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


def _test_01() -> None:
    import numpy as np

    model = EvoSteadyState(
        -8,
        8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda: np.random.normal(0, 0.15),
        pop_size_parent=1,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=512)
    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    _test_01()
