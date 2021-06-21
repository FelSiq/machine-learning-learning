import collections
import os

import pandas as pd
import numpy as np


def add_pad(X):
    zeros = np.zeros(X.shape[0], dtype=float)
    return np.column_stack((zeros, X, zeros))


def get_data(test_frac: float = 0.1, nrows=None):
    assert 1.0 > test_frac > 0.0
    dtypes = collections.defaultdict(float)
    dtypes["category"] = str

    datapath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/time_series_data.csv"
    )

    data = pd.read_csv(
        datapath,
        index_col=0,
        header=0,
        dtype=dtypes,
        nrows=nrows,
    )
    data = data.sample(frac=1, random_state=16, axis="index").reset_index(drop=True)

    test_size = int(data.shape[0] * test_frac)

    X_train = data.iloc[test_size:, :-1].values
    X_test = data.iloc[:test_size, :-1].values

    X_train = add_pad(X_train)
    X_test = add_pad(X_test)

    return X_train, X_test
