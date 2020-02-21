"""Evolutionary Programming class."""
import evobatch


class EvoProg(evobatch.EvoBatch):
    """Evolutionary Programming evolutionary algorithm.

    +------------------------------+----------------+
    | Algorithm characteristic:    | Value:         |
    +------------------------------+----------------+
    | Overlapping population       | No             |
    +------------------------------+----------------+
    | Parent population size       | m > 0          |
    +------------------------------+----------------+
    | Offspring population size    | n > 0          |
    +------------------------------+----------------+
    | Parent selection scheme      | Deterministic  |
    +------------------------------+----------------+
    | Offspring selection scheme   | Truncation     |
    +------------------------------+----------------+
    | Reproduction                 | Asexual        |
    +------------------------------+----------------+
    | Mutation                     | Yes            |
    +------------------------------+----------------+
    | Crossover                    | No             |
    +------------------------------+----------------+
    """

    def __init__(self, *args, **kwargs):
        """."""
        kwargs.setdefault("pop_size_parent", 1)
        kwargs.setdefault("pop_size_parent", 256)

        super().__init__(
            overlapping_pops=False,
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
        mutation_delta_func=lambda: np.random.normal(0, 0.15),
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=2048)
    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    _test()
