"""Implements Random Forest Classifier algorithm."""
import typing as t

import numpy as np

import dt


class RandomForestClassifier:
    """Ensemble of Decision Tree Classifiers."""

    def __init__(self,
                 n_trees: int = 256,
                 max_depth: t.Optional[int] = None,
                 stride_frac: float = 0.1,
                 min_impurity: float = 1e-7):
        """Init a Decision Tree Classifier model."""
        if n_trees <= 0:
            raise ValueError(
                "'n_trees' must be positive (got {}.)".format(n_trees))

        self.max_depth = np.inf if max_depth is None else max_depth
        self.min_impurity = min_impurity
        self.n_trees = n_trees
        self._estimators = []  # type: t.List[dt.DecisionTreeClassifier]
        self._tree_type = dt.DecisionTreeNumeric
        self.oob_accuracy = -1.0
        self.stride_frac = stride_frac

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            attr_sample_frac: t.Optional[float] = None,
            calc_oob_accuracy: bool = True,
            random_state: t.Optional[int] = None,
    ) -> "RandomForestClassifier":
        """Fit an ensemble of Decision Tree Classifiers."""
        if attr_sample_frac is None:
            attr_sample_frac = np.sqrt(X.shape[1])

        if calc_oob_accuracy:
            inst_oob_preds = np.full((y.size, self.n_trees), None)

        _all_inds = np.arange(y.size, dtype=int)

        if random_state is not None:
            np.random.seed(random_state)

        bootstrap_inds = np.random.randint(y.size, size=(self.n_trees, y.size))

        for tree_id in np.arange(self.n_trees):
            cur_bootstrap_inds = bootstrap_inds[tree_id, :]

            X_sample = X[cur_bootstrap_inds, :]
            y_sample = y[cur_bootstrap_inds]

            self._estimators.append(
                self._tree_type(
                    max_depth=self.max_depth,
                    stride_frac=self.stride_frac,
                    min_impurity=self.min_impurity).fit(
                        X=X_sample,
                        y=y_sample,
                        attr_sample_frac=attr_sample_frac,
                        random_state=random_state))

            if calc_oob_accuracy:
                oob_inds = np.delete(_all_inds, cur_bootstrap_inds)
                inst_oob_preds[oob_inds, tree_id] = self._estimators[
                    -1].predict(X[oob_inds, :])

        if calc_oob_accuracy:
            inst_was_oob = np.any(inst_oob_preds != None, axis=1)

            oob_inst_classes = np.array([
                self._predict_inst(
                    inst=None, preds=oob_pred[oob_pred != None])
                for oob_pred, was_oob in zip(inst_oob_preds, inst_was_oob)
                if was_oob
            ])

            self.oob_accuracy = np.mean(oob_inst_classes == y[inst_was_oob])

        return self

    def _predict_inst(self,
                      inst: np.ndarray,
                      preds: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Use the tree ensemble to predict a single instance class."""
        classes, freqs = np.unique(
            [tree.predict(inst)
             for tree in self._estimators] if preds is None else preds,
            return_counts=True)

        return classes[np.argmax(freqs)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use each Decision Tree Classifier make predictions."""
        return np.apply_along_axis(func1d=self._predict_inst, arr=X, axis=0)


def _test() -> None:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier as SklRFC
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt

    iris = load_iris()
    n_trees_opts = np.arange(100, 300 + 1, 25)

    accs_rf_50 = np.zeros(n_trees_opts.size, dtype=float)
    accs_rf_75 = np.zeros(n_trees_opts.size, dtype=float)
    accs_bagging = np.zeros(n_trees_opts.size, dtype=float)
    accs_sklearn = np.zeros(n_trees_opts.size, dtype=float)

    for opt_ind, n_trees in enumerate(n_trees_opts):
        model = RandomForestClassifier(n_trees=n_trees, stride_frac=0.2)
        model_skl = SklRFC(n_estimators=n_trees, oob_score=True)

        model.fit(iris.data, iris.target, attr_sample_frac=0.50)
        accs_rf_50[opt_ind] = model.oob_accuracy

        model.fit(iris.data, iris.target, attr_sample_frac=0.75)
        accs_rf_75[opt_ind] = model.oob_accuracy

        model.fit(iris.data, iris.target, attr_sample_frac=1.0)
        accs_bagging[opt_ind] = model.oob_accuracy

        model_skl.fit(iris.data, iris.target)
        accs_sklearn[opt_ind] = model_skl.oob_score_

        print("\r{:.2f}%".format(100 * opt_ind / n_trees_opts.size), end="")

    baseline_dt = 0.0
    n_splits = 10
    splitter = StratifiedKFold(n_splits=n_splits)
    model = dt.DecisionTreeNumeric()

    for inds_train, inds_test in splitter.split(iris.data, iris.target):
        model.fit(iris.data[inds_train, :], iris.target[inds_train])
        baseline_dt += np.mean(model.predict(iris.data[inds_test, :]) == iris.target[inds_test])

    baseline_dt /= n_splits

    plt.title("Out-of-bag (OOB) accuracy")
    plt.plot(n_trees_opts, accs_rf_75, label="RF (75% attr sample)")
    plt.plot(n_trees_opts, accs_rf_50, "--", label="RF (50% attr sample)")
    plt.plot(
        n_trees_opts, accs_bagging, "o-", label="bagging (100% attr sample)")
    plt.plot(n_trees_opts, accs_sklearn, ".", label="sklearn RF")

    _, freqs = np.unique(iris.target, return_counts=True)
    baseline_rd = np.max(freqs) / iris.target.size

    plt.hlines(
        y=baseline_rd,
        xmin=n_trees_opts[0],
        xmax=n_trees_opts[-1],
        label="Baseline random",
        linestyles="dotted")

    plt.hlines(
        y=baseline_dt,
        xmin=n_trees_opts[0],
        xmax=n_trees_opts[-1],
        label="Baseline single DT",
        linestyles="dashdot")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test()
