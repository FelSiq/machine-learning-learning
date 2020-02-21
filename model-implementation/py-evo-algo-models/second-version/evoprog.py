"""Evolutionary Programming class."""
import evobatch


class EvoProg(evobatch.EvoBatch):
    """Evolutionary Programming evolutionary algorithm.

    +------------------------------+----------------+
    | Algorithm characteristic:    | Value:         |
    +------------------------------+----------------+
    | Overlapping population       | Yes            |
    +------------------------------+----------------+
    | Parent population size       | m > 0          |
    +------------------------------+----------------+
    | Offspring population size    | m              |
    +------------------------------+----------------+
    | Parent selection scheme      | Deterministic  |
    +------------------------------+----------------+
    | Offspring selection scheme   | Truncation     |
    +------------------------------+----------------+
    | Merge populations to select  | True           |
    +------------------------------+----------------+
    | Reproduction                 | Asexual        |
    +------------------------------+----------------+
    | Mutation                     | Yes            |
    +------------------------------+----------------+
    | Crossover                    | No             |
    +------------------------------+----------------+
    """

    def __init__(self, *args, **kwargs):
        """Init a evolutionary programming model."""
        super().__init__(
            overlapping_pops=True,
            merge_populations=True,
            pop_size_offspring=None,
            selection_parent="deterministic",
            selection_target="truncation",
            *args,
            **kwargs)

        self._alg_name = "Evolutionary Programming"


def _test() -> None:
    import numpy as np

    model = EvoProg(
        -8,
        8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda: np.random.normal(0, 0.1),
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=96)
    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    _test()
