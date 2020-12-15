# -*- coding: utf8 -*-
"""Experimenting with metalearning algorithm recomendation.

The purpose of this module is just learn more about this
topic. There's no intention of building up something really
useful.
"""
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import spearmanr


class MlRecommender:
    """Class for testing alg. recommendation with metalearning."""

    def __init__(self,
                 filepath=None,
                 k=5,
                 perf_columns=None,
                 ignore_columns=None):
        """Uses k-NN to rank base learners algorithms."""
        self._pred_model = MultiOutputRegressor(KNeighborsRegressor())
        self._rank_baseline = None
        self.x_attr = None
        self.y_attr = None
        self.neighbor_num = k
        self.baseline_alg = None

        if filepath is not None:
            if perf_columns is None:
                raise ValueError("perf_columns must be specified \
                    alongside filepath parameter.")

            self.load_metadata(
                filepath=filepath,
                perf_columns=perf_columns,
                ignore_columns=ignore_columns)

    @classmethod
    def _average_ranking(cls, rankings):
        """Calculate average ranking."""
        if not isinstance(rankings, np.array):
            rankings = np.array(rankings)

        return rankings.mean(axis=0)

    def _get_baseline(self):
        """Get Expected value by random recommendation as
        baseline value.
        """
        self._rank_baseline = self.y_attr.mean(axis=0)

    def _fill_na_knn(self):
        """Fill missing values using k-NN values."""
        self.x_attr[np.isnan(self.x_attr)] = 0

    def load_metadata(self, filepath, perf_columns, ignore_columns=None):
        """Load metadata with rank or performance of
        base learners columns specified via per_columns
        parameter.
        """
        metadata = pd.read_csv(filepath)

        if ignore_columns is not None:
            metadata.drop(
                metadata.columns[ignore_columns], axis=1, inplace=True)

        self.x_attr = metadata.drop(
            metadata.columns[perf_columns], axis=1).values
        self.y_attr = metadata.iloc[:, perf_columns].values
        self.baseline_alg = metadata.columns[perf_columns].values

        self._fill_na_knn()

    def predict(self, query):
        """."""
        if self.x_attr is None:
            raise Exception("first call \"load_metadata\" method.")

        return self._pred_model.predict(query)

    def loocv_validate(self):
        """."""
        y_attr_size = len(self.y_attr)
        train_index = list(range(y_attr_size))
        output = np.zeros((y_attr_size, 2))

        for test_index in range(y_attr_size):
            train_index.remove(test_index)

            self._pred_model.fit(
                X=self.x_attr[train_index, :], y=self.y_attr[train_index, :])

            test_ranking = self._pred_model.predict(
                X=self.x_attr[test_index, :].reshape(1, -1)).flatten()

            output[test_index, :] = spearmanr(test_ranking,
                                              self.y_attr[test_index, :])

            train_index.append(test_index)

        return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage:", sys.argv[0], "<filepath>",
              "<perf_columns_comma_separated>",
              "[ignored_columns_comma_separated]")
        exit(1)

    PERF_COLUMNS = list(map(int, sys.argv[2].strip().split(",")))

    try:
        IGNORED_COLUMNS = list(map(int, sys.argv[3].strip().split(",")))

    except TypeError:
        IGNORED_COLUMNS = None

    rec = MlRecommender(
        filepath=sys.argv[1],
        perf_columns=PERF_COLUMNS,
        ignore_columns=IGNORED_COLUMNS)

    res = rec.loocv_validate()

    print(rec.x_attr.shape)
    print("baseline algorithms:", rec.baseline_alg)
    print("Results:")
    print(res)
