"""Steady state evolutionary algorithm."""
import evobatch


class EvoSteadyState(evobatch.EvoBatch):
    """Steady State (batch size = 1) evolutionary algorithm.

    +------------------------------+----------------+
    | Algorithm characteristic:    | Value:         |
    +------------------------------+----------------+
    | Overlapping population       | Yes            |
    +------------------------------+----------------+
    | Parent population size       | m > 0          |
    +------------------------------+----------------+
    | Offspring population size    | 1              |
    +------------------------------+----------------+
    | Parent selection scheme      | Any            |
    +------------------------------+----------------+
    | Offspring selection scheme   | Any            |
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
        """Init a Steady State (batch size = 1) evolutionary model."""
        super().__init__(
            overlapping_pops=True,
            merge_populations=False,
            pop_size_offspring=1,
            reproduction="asexual",
            reproduction_func=None,
            reproduction_func_args=None,
            *args,
            **kwargs)

        self._alg_name = "Steady State/Incremental"


def _test_01() -> None:
    import numpy as np

    model = EvoSteadyState(
        -8,
        8,
        fitness_func=
        lambda inst: np.sum(inst[0] * np.sin(inst[1])) if abs(inst[0]) < 7 else 0.0,
        mutation_delta_func=lambda: np.random.normal(0, 0.15),
        selection_parent="tournament",
        selection_target="uniform",
        selection_parent_args={"size": 2},
        pop_size_parent=16,
        gen_range_low=[-2.5, -8],
        gen_range_high=[2.5, 8],
        gene_num=2,
        gen_num=48)
    model.run(verbose=True, plot=True, pause=0.01)
    print(model)


if __name__ == "__main__":
    _test_01()
