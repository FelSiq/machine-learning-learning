# -*- coding: utf8 -*-
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score


class MlRecommender:
    def __init__(self, filepath=None, k=5,
                 perf_columns=None, ignore_columns=None):
        """Uses k-NN to rank base learners algorithms."""
        self._pred_model = MultiOutputRegressor(KNeighborsRegressor())
        self._rank_baseline = None
        self.X = None
        self.y = None
        self.k = k

        if filepath is not None:
            if perf_columns is None:
                raise ValueError("perf_columns must be specified \
                    alongside filepath parameter.")

            self.load_metadata(
                    filepath=filepath,
                    perf_columns=perf_columns,
                    ignore_columns=ignore_columns)

    def _average_ranking(self, rankings):
        """Calculate average ranking."""
        if type(rankings) is not np.array:
            rankings = np.array(rankings)

        return rankings.mean(axis=0)

    def _get_baseline(self):
        """Get Expected value by random recommendation as
        baseline value.
        """
        self._rank_baseline = self.y.mean(axis=0)

    def _fill_na_knn(self):
        """Fill missing values using k-NN values."""
        """To do."""
        self.X[np.isnan(self.X)] = 0

    def load_metadata(self, filepath, perf_columns, ignore_columns=None):
        """Load metadata with rank or performance of
        base learners columns specified via per_columns
        parameter.
        """
        metadata = pd.read_csv(filepath)

        if ignore_columns is not None:
            metadata.drop(metadata.columns[ignore_columns],
                          axis=1, inplace=True)

        self.X = metadata.drop(
                metadata.columns[perf_columns], axis=1).values
        self.y = metadata.iloc[:, perf_columns].values

        self._fill_na_knn()
        self._pred_model.fit(self.X, self.y)

        self._get_baseline()

    def predict(self, query):
        """."""
        if self.metadata is None:
            raise Exception("first call \"load_metadata\" method.")

        return self._pred_model.predict(query)

    def validate(self, score_metrics=None, cv=10):
        """."""
        if score_metrics is None:
            score_metrics = "explained_variance"

        scores = cross_val_score(
                self._pred_model,
                self.X,
                self.y,
                scoring=score_metrics,
                cv=cv)

        return scores

    def plot(self, performance, baseline):
        """."""
        pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage:", sys.argv[0], "<filepath>",
              "<perf_columns_comma_separated>",
              "[ignored_columns_comma_separated]")
        exit(1)

    perf_columns = list(map(int, sys.argv[2].strip().split(",")))

    try:
        ignored_columns = list(map(int, sys.argv[3].strip().split(",")))

    except Exception:
        ignored_columns = None

    rec = MlRecommender(
        filepath=sys.argv[1],
        perf_columns=perf_columns,
        ignore_columns=ignored_columns)

    res = rec.validate()

    print(res)
