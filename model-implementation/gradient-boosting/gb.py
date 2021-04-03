import typing as t
import functools

import numpy as np
import sklearn.tree
import sklearn.base
import tqdm.auto

_LearnerType = t.Union[
    sklearn.tree.DecisionTreeClassifier, sklearn.tree.DecisionTreeRegressor
]


class _GradientBoostingBase(sklearn.base.BaseEstimator):
    def __init__(
        self,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        *args,
        **kwargs,
    ):
        assert 0 < float(learning_rate) <= 1.0
        assert int(n_estimators) > 0

        self.learning_rate = float(learning_rate)

        self.n_estimators = int(n_estimators)
        self.estimators = []  # type: t.List[_LearnerType]

        self._estimator_gen = functools.partial(
            sklearn.tree.DecisionTreeRegressor, max_depth=max_depth
        )
        self._baseline = np.nan

    def _check_X_y(self, X: np.ndarray, y: t.Optional[np.ndarray] = None):
        X = np.asarray(X)

        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        if y is not None:
            if isinstance(self, GradientBoostingClassifier):
                cls = set(np.unique(y))
                assert len(cls) == 2 and not cls.symmetric_difference({0, 1})
                y = np.asarray(y, dtype=int).ravel()

            else:
                y = np.asfarray(y).ravel()

            return X, y

        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = self._check_X_y(X, y)

        self.estimators = []
        self._baseline = self._calc_baseline(y)
        cur_pred = self._baseline
        pseudo_resid = y - self._baseline

        for _ in tqdm.auto.tqdm(np.arange(self.n_estimators)):
            estimator = self._estimator_gen()
            estimator.fit(X, pseudo_resid)

            region_ids = estimator.apply(X)
            pred_pseudo_resid = estimator.predict(X)

            cur_pred = cur_pred + self.learning_rate * pred_pseudo_resid
            pseudo_resid = y - cur_pred
            self.estimators.append(estimator)

        return self

    def predict(self, X):
        X = self._check_X_y(X)
        preds = np.full(X.shape[0], fill_value=self._baseline)

        for estimator in self.estimators:
            preds += self.learning_rate * estimator.predict(X)

        return preds

    def _calc_baseline(self, y: np.ndarray) -> float:
        return float(np.mean(y))


class GradientBoostingRegressor(_GradientBoostingBase):
    def __init__(self, *args, **kwargs):
        super(GradientBoostingRegressor, self).__init__(*args, **kwargs)


class GradientBoostingClassifier(_GradientBoostingBase):
    def __init__(self, *args, **kwargs):
        super(GradientBoostingClassifier, self).__init__(*args, **kwargs)


def _test():
    import sklearn.ensemble
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics

    gb_reg = GradientBoostingRegressor()
    sk_reg = sklearn.ensemble.GradientBoostingRegressor()

    X, y = sklearn.datasets.load_boston(return_X_y=True)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2
    )

    gb_reg.fit(X_train, y_train)
    sk_reg.fit(X_train, y_train)

    gb_y_pred = gb_reg.predict(X_eval)
    sk_y_pred = sk_reg.predict(X_eval)

    gb_rmse = sklearn.metrics.mean_squared_error(gb_y_pred, y_eval, squared=False)
    sk_rmse = sklearn.metrics.mean_squared_error(sk_y_pred, y_eval, squared=False)

    print(f"My rmse: {gb_rmse:.4f}")
    print(f"Sk rmse: {sk_rmse:.4f}")


if __name__ == "__main__":
    _test()
