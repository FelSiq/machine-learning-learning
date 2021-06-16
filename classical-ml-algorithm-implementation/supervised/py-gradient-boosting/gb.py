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
        self.estimator_layers = []  # type: t.List[_LearnerType]

        self._estimator_gen = functools.partial(
            sklearn.tree.DecisionTreeRegressor,
            max_depth=max_depth,
            criterion="mse",
            *args,
            **kwargs,
        )
        self._baseline = np.nan
        self._n_labels = 0
        self._encoder = sklearn.preprocessing.OneHotEncoder(
            drop="if_binary", sparse=False, dtype=float
        )

    def _check_X_y(self, X: np.ndarray, y: t.Optional[np.ndarray] = None):
        X = np.asarray(X)

        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        if y is not None:
            assert y.ndim <= 2

            if isinstance(self, GradientBoostingClassifier):
                y = self._encoder.fit_transform(
                    y if y.ndim == 2 else np.expand_dims(y, axis=1)
                )
                assert len(self._encoder.categories_) >= 1

            else:
                y = np.asfarray(y).reshape(-1, 1)

            assert X.shape[0] == y.shape[0]
            assert X.ndim == y.ndim == 2

            return X, y

        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = self._check_X_y(X, y)
        self._prepare_fit()
        self._n_labels = y.shape[1]

        self.estimator_layers = []
        self._baseline = self._calc_baseline(y)
        cur_preds = np.tile(self._baseline, (y.shape[0], 1))

        if isinstance(self, GradientBoostingClassifier):
            cur_preds_probs = self._log_odds_to_prob(cur_preds)

        else:
            cur_preds_probs = np.zeros((1, self._n_labels))

        ref = self._calc_ref(y)
        pseudo_resid = ref - self._baseline

        for _ in tqdm.auto.tqdm(np.arange(self.n_estimators)):
            layer = []

            for j in np.arange(self._n_labels):
                estimator = self._estimator_gen()
                estimator.fit(X, pseudo_resid[:, j])

                # Units: the same as 'y'
                preds_pseudo_resid = estimator.predict(X)

                # Units: the same as 'self._baseline' or 'cur_preds
                preds = self._cast_pseudo_resid_to_preds(
                    X, estimator, preds_pseudo_resid, cur_preds_probs[:, j]
                )

                cur_preds[:, j] += self.learning_rate * preds
                layer.append(estimator)

            pseudo_resid = ref - cur_preds
            self.estimator_layers.append(layer)

            if isinstance(self, GradientBoostingClassifier):
                cur_preds_probs = self._log_odds_to_prob(cur_preds)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X_y(X)
        preds = np.tile(self._baseline, (X.shape[0], 1))

        for estimator_layer in self.estimator_layers:
            for j, estimator in enumerate(estimator_layer):
                preds_pseudo_resid = estimator.predict(X)
                cur_preds = self._cast_pseudo_resid_to_preds(
                    X, estimator, preds_pseudo_resid, train=False
                )
                preds[:, j] += self.learning_rate * cur_preds

        return np.squeeze(self._prepare_output(preds))

    def _prepare_fit(self):
        pass

    def _calc_ref(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _cast_pseudo_resid_to_preds(
        self,
        X: np.ndarray,
        estimator: _LearnerType,
        pseudo_resids: t.Optional[np.ndarray] = None,
        cur_preds_probs: t.Optional[np.ndarray] = None,
        train: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def _prepare_output(self, preds: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def self_calc_baseline(cls, y: np.ndarray) -> float:
        raise NotImplementedError


class GradientBoostingRegressor(_GradientBoostingBase):
    def _calc_ref(self, y: np.ndarray) -> np.ndarray:
        return y

    def _prepare_output(self, preds: np.ndarray) -> np.ndarray:
        return preds

    def _cast_pseudo_resid_to_preds(
        self,
        X: np.ndarray,
        estimator: _LearnerType,
        pseudo_resids: t.Optional[np.ndarray] = None,
        cur_preds_probs: t.Optional[np.ndarray] = None,
        train: bool = True,
    ) -> np.ndarray:
        return np.asfarray(pseudo_resids)

    @classmethod
    def _calc_baseline(cls, y: np.ndarray) -> float:
        cls_prob = float(np.mean(y))
        return cls_prob


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
        cur_preds_probs: t.Optional[np.ndarray] = None,
        train: bool = True,
    ) -> np.ndarray:
        leaf_ids = estimator.apply(X)

        if train:
            assert pseudo_resids is not None and cur_preds_probs is not None
            _pseudo_resids = np.asfarray(pseudo_resids)
            _cur_preds_probs = np.asfarray(cur_preds_probs)
            self._register_estimator_log_odds_by_leaf(
                leaf_ids, estimator, _pseudo_resids, _cur_preds_probs
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
        cur_preds_probs: np.ndarray,
    ) -> None:
        _id_est = id(estimator)

        assert _id_est not in self._estimators_log_odds_by_leaf

        all_leaf_ids, region_ids = np.unique(leaf_ids, return_inverse=True)

        total_var_by_leaf, total_resid_by_leaf = np.zeros((2, estimator.get_n_leaves()))

        np.add.at(total_resid_by_leaf, region_ids, pseudo_resids)
        np.add.at(
            total_var_by_leaf, region_ids, cur_preds_probs * (1.0 - cur_preds_probs)
        )

        log_odds_by_leaf = np.asfarray(total_resid_by_leaf / total_var_by_leaf)

        self._estimators_log_odds_by_leaf[_id_est] = log_odds_by_leaf
        self._estimators_leaf_id_translator[_id_est] = all_leaf_ids[:, np.newaxis]

    def _log_odds_to_prob(self, log_odds: np.ndarray) -> t.Union[float, np.ndarray]:
        # NOTE: 'log_odds' is multiplied by -1 to simulate the transformation
        # from the cross entropy loss (which requires a gradient descent to minimize)
        # to the maximum likelihood estimation of a multinomial distribution (which uses
        # gradient ascent and, therefore, is complatible with our implementation of both
        # regression and binary classification). I don't know if there is a more both
        # appropriate and generic way to make this transformation in this context.
        probs = (
            self._softmax(-log_odds) if self._n_labels > 1 else self._sigmoid(log_odds)
        )
        return probs

    def _prepare_output(self, log_odds: np.ndarray):
        return self._log_odds_to_prob(log_odds)

    @staticmethod
    def _sigmoid(log_odds: t.Union[float, np.ndarray]) -> t.Union[float, np.ndarray]:
        return 1.0 / (1.0 + np.exp(-log_odds))

    @staticmethod
    def _softmax(log_odds: np.ndarray) -> np.ndarray:
        exp_shifted = np.exp(log_odds - np.max(log_odds, axis=-1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

    def _calc_ref(self, y: np.ndarray) -> np.ndarray:
        inst_log_odds = np.where(y, self._baseline, -self._baseline)
        return np.asfarray(inst_log_odds)

    @classmethod
    def _calc_baseline(cls, y: np.ndarray) -> t.Union[float, np.ndarray]:
        cls_prob = np.mean(y, axis=0)
        cls_odds = cls_prob / (1.0 - cls_prob)
        cls_log_odds = np.log(cls_odds)
        return cls_log_odds


def _test():
    import sklearn.ensemble
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics

    regression = False
    binary_class = True

    if regression:
        gb = GradientBoostingRegressor()
        sk = sklearn.ensemble.GradientBoostingRegressor()
        X, y = sklearn.datasets.load_boston(return_X_y=True)
    else:
        gb = GradientBoostingClassifier()
        sk = sklearn.ensemble.GradientBoostingClassifier()

        if binary_class:
            X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        else:

            X, y = sklearn.datasets.load_iris(return_X_y=True)

    splitter = sklearn.model_selection.KFold(n_splits=10, shuffle=True)

    gb_perf = sk_perf = 0.0

    for train_inds, eval_inds in splitter.split(X, y):
        X_train, X_eval = X[train_inds, :], X[eval_inds, :]
        y_train, y_eval = y[train_inds], y[eval_inds]

        gb.fit(X_train, y_train)
        sk.fit(X_train, y_train)

        gb_y_pred = gb.predict(X_eval)
        sk_y_pred = sk.predict(X_eval)

        if regression:
            gb_perf += sklearn.metrics.mean_squared_error(
                gb_y_pred, y_eval, squared=False
            )
            sk_perf += sklearn.metrics.mean_squared_error(
                sk_y_pred, y_eval, squared=False
            )

        else:
            if gb_y_pred.ndim == 1:
                gb_y_pred = gb_y_pred > 0.5

            else:
                gb_y_pred = gb_y_pred.argmax(axis=1)

            gb_y_pred = gb_y_pred.astype(int, copy=False)

            gb_perf += sklearn.metrics.accuracy_score(gb_y_pred, y_eval)
            sk_perf += sklearn.metrics.accuracy_score(sk_y_pred, y_eval)

    print(f"My perf: {gb_perf / 10:.4f}")
    print(f"Sk perf: {sk_perf / 10:.4f}")


if __name__ == "__main__":
    _test()
