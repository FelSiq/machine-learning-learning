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
            sklearn.tree.DecisionTreeRegressor,
            max_depth=max_depth,
            criterion="mse",
            *args,
            **kwargs,
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
        self._prepare_fit()

        self.estimators = []
        self._baseline = self._calc_baseline(y)
        cur_preds = np.full(y.size, fill_value=self._baseline)

        ref = self._calc_ref(y)
        pseudo_resid = ref - self._baseline

        for _ in tqdm.auto.tqdm(np.arange(self.n_estimators)):
            estimator = self._estimator_gen()
            estimator.fit(X, pseudo_resid)

            # Units: the same as 'y'
            preds_pseudo_resid = estimator.predict(X)

            # Units: the same as 'self._baseline' or 'cur_preds
            preds = self._cast_pseudo_resid_to_preds(
                X, estimator, preds_pseudo_resid, cur_preds
            )

            cur_preds += self.learning_rate * preds
            pseudo_resid = ref - cur_preds

            self.estimators.append(estimator)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X_y(X)
        preds = np.full(X.shape[0], fill_value=self._baseline)

        for estimator in self.estimators:
            preds_pseudo_resid = estimator.predict(X)
            cur_preds = self._cast_pseudo_resid_to_preds(
                X, estimator, preds_pseudo_resid, train=False
            )
            preds += self.learning_rate * cur_preds

        return self._prepare_output(preds)

    def _prepare_fit(self):
        pass

    def _calc_ref(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _cast_pseudo_resid_to_preds(
        self,
        X: np.ndarray,
        estimator: _LearnerType,
        pseudo_resids: t.Optional[np.ndarray] = None,
        cur_preds: t.Optional[np.ndarray] = None,
        train: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def self_calc_baseline(cls, y: np.ndarray) -> float:
        raise NotImplementedError

    @classmethod
    def _prepare_output(cls, preds: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientBoostingRegressor(_GradientBoostingBase):
    def _calc_ref(self, y: np.ndarray) -> np.ndarray:
        return y

    def _cast_pseudo_resid_to_preds(
        self,
        X: np.ndarray,
        estimator: _LearnerType,
        pseudo_resids: t.Optional[np.ndarray] = None,
        cur_preds: t.Optional[np.ndarray] = None,
        train: bool = True,
    ) -> np.ndarray:
        return np.asfarray(pseudo_resids)

    @classmethod
    def _calc_baseline(cls, y: np.ndarray) -> float:
        cls_prob = float(np.mean(y))
        return cls_prob

    @classmethod
    def _prepare_output(cls, preds: np.ndarray) -> np.ndarray:
        return preds


class GradientBoostingClassifier(_GradientBoostingBase):
    def __init__(self, *args, **kwargs):
        super(GradientBoostingClassifier, self).__init__(*args, **kwargs)
        self._estimators_log_odds_by_leaf = dict()
        self._estimators_leaf_id_translator = dict()

    def _prepare_fit(self):
        self._estimators_log_odds_by_leaf = dict()
        self._estimators_leaf_id_translator = dict()

    def _cast_pseudo_resid_to_preds(
        self,
        X: np.ndarray,
        estimator: _LearnerType,
        pseudo_resids: t.Optional[np.ndarray] = None,
        cur_preds: t.Optional[np.ndarray] = None,
        train: bool = True,
    ) -> np.ndarray:
        leaf_ids = estimator.apply(X)

        if train:
            assert pseudo_resids is not None and cur_preds is not None
            _pseudo_resids = np.asfarray(pseudo_resids)
            _cur_preds = np.asfarray(cur_preds)
            self._register_estimator_log_odds_by_leaf(
                leaf_ids, estimator, _pseudo_resids, _cur_preds
            )

        _id_est = id(estimator)
        all_leaf_ids = self._estimators_leaf_id_translator[_id_est]
        log_odds_by_leaf = self._estimators_log_odds_by_leaf[_id_est]

        region_ids = np.argmax(leaf_ids == all_leaf_ids, axis=0)
        pred_log_odds = log_odds_by_leaf[region_ids]

        return pred_log_odds

    def _register_estimator_log_odds_by_leaf(
        self,
        leaf_ids: np.ndarray,
        estimator: _LearnerType,
        pseudo_resids: np.ndarray,
        cur_preds: np.ndarray,
    ) -> None:
        _id_est = id(estimator)

        assert _id_est not in self._estimators_log_odds_by_leaf

        all_leaf_ids, region_ids = np.unique(leaf_ids, return_inverse=True)

        total_var_by_leaf, total_resid_by_leaf = np.zeros((2, estimator.get_n_leaves()))
        cur_preds_probs = self._sigmoid(cur_preds)

        np.add.at(total_resid_by_leaf, region_ids, pseudo_resids)
        np.add.at(
            total_var_by_leaf, region_ids, cur_preds_probs * (1.0 - cur_preds_probs)
        )

        log_odds_by_leaf = np.asfarray(total_resid_by_leaf / total_var_by_leaf)

        self._estimators_log_odds_by_leaf[_id_est] = log_odds_by_leaf
        self._estimators_leaf_id_translator[_id_est] = all_leaf_ids[:, np.newaxis]

    @staticmethod
    def _sigmoid(log_odds: t.Union[float, np.ndarray]) -> t.Union[float, np.ndarray]:
        return 1.0 / (1.0 + np.exp(-log_odds))

    def _calc_ref(self, y: np.ndarray) -> np.ndarray:
        inst_log_odds = np.where(y, self._baseline, -self._baseline)
        return np.asfarray(inst_log_odds)

    @classmethod
    def _prepare_output(
        cls,
        log_odds: np.ndarray,
    ):
        probs = cls._sigmoid(log_odds)
        return probs

    @classmethod
    def _calc_baseline(cls, y: np.ndarray) -> float:
        cls_prob = np.mean(y)
        cls_odds = cls_prob / (1.0 - cls_prob)
        cls_log_odds = float(np.log(cls_odds))
        return cls_log_odds


def _test():
    import sklearn.ensemble
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics

    regression = False

    if regression:
        gb = GradientBoostingRegressor()
        sk = sklearn.ensemble.GradientBoostingRegressor()
        X, y = sklearn.datasets.load_boston(return_X_y=True)
    else:
        gb = GradientBoostingClassifier()
        sk = sklearn.ensemble.GradientBoostingClassifier()
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2
    )

    gb.fit(X_train, y_train)
    sk.fit(X_train, y_train)

    gb_y_pred = gb.predict(X_eval)
    sk_y_pred = sk.predict(X_eval)

    if regression:
        gb_rmse = sklearn.metrics.mean_squared_error(gb_y_pred, y_eval, squared=False)
        sk_rmse = sklearn.metrics.mean_squared_error(sk_y_pred, y_eval, squared=False)

        print(f"My rmse: {gb_rmse:.4f}")
        print(f"Sk rmse: {sk_rmse:.4f}")

    else:
        gb_y_pred = (gb_y_pred > 0.5).astype(int)
        gb_acc = sklearn.metrics.accuracy_score(gb_y_pred, y_eval)
        sk_acc = sklearn.metrics.accuracy_score(sk_y_pred, y_eval)

        print(f"My acc: {gb_acc:.4f}")
        print(f"Sk acc: {sk_acc:.4f}")


if __name__ == "__main__":
    _test()
