import typing as t

import numpy as np


class PCA:
    def __init__(self, n_components: t.Union[int, float]):
        assert n_components > 0.0

        self.n_components = n_components

    def fit(self, X):
        pass

    def transform(self, X):
        pass



def _test():
    pass


if __name__ == "__main__":
    _test()
