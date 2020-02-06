"""Implements Random Forest Classifier algorithm."""
import typing as t

import numpy as np

import dt


class RandomForestClassifier:
    """."""

    def __init__(self,
                 n_trees: int = 256,
                 max_depth: t.Optional[int] = None,
                 min_impurity: float = 0.0):
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

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            attr_sample_frac: t.Optional[float] = 0.75,
            calc_oob_accuracy: bool = True,
            random_state: t.Optional[int] = None,
    ) -> "RandomForestClassifier":
        """Fit an ensemble of Decision Tree Classifiers."""
        if attr_sample_frac is None:
            attr_sample_frac = np.sqrt(X.shape[1])

        if random_state is not None:
            np.random.seed(random_state)

        if calc_oob_accuracy:
            inst_oob_preds = np.full((y.size, self.n_trees), None)

        _all_inds = np.arange(y.size, dtype=int)

        for tree_id in np.arange(self.n_trees):
            bootstrap_inds = np.random.randint(y.size, size=y.size)

            X_sample = X[bootstrap_inds, :]
            y_sample = y[bootstrap_inds]

            self._estimators.append(
                self._tree_type(
                    max_depth=self.max_depth,
                    min_impurity=self.min_impurity).fit(
                        X=X_sample,
                        y=y_sample,
                        attr_sample_frac=attr_sample_frac,
                        random_state=random_state))

            if calc_oob_accuracy:
                oob_inds = np.delete(_all_inds, bootstrap_inds)
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
    import matplotlib.pyplot as plt

    iris = load_iris()
    n_trees_opts = np.arange(1, 301, 25)

    accs_rf_75 = np.zeros(n_trees_opts.size, dtype=float)
    accs_rf_50 = np.zeros(n_trees_opts.size, dtype=float)
    accs_bagging = np.zeros(n_trees_opts.size, dtype=float)
    accs_sklearn = np.zeros(n_trees_opts.size, dtype=float)

    for opt_ind, n_trees in enumerate(n_trees_opts):
        model = RandomForestClassifier(n_trees=n_trees)
        model_skl = SklRFC(n_estimators=n_trees, oob_score=True)

        model.fit(iris.data, iris.target, attr_sample_frac=0.75)
        accs_rf_75[opt_ind] = model.oob_accuracy

        model.fit(iris.data, iris.target, attr_sample_frac=0.50)
        accs_rf_50[opt_ind] = model.oob_accuracy

        model.fit(iris.data, iris.target, attr_sample_frac=1.0)
        accs_bagging[opt_ind] = model.oob_accuracy

        model_skl.fit(iris.data, iris.target)
        accs_sklearn[opt_ind] = model_skl.oob_score_

        print("\r{:.2f}%".format(100 * opt_ind / n_trees_opts.size), end="")

    plt.title("Out-of-bag (OOB) accuracy")
    plt.plot(n_trees_opts, accs_rf_75, label="RF (75% attr sample)")
    plt.plot(n_trees_opts, accs_rf_50, "--", label="RF (50% attr sample)")
    plt.plot(
        n_trees_opts, accs_bagging, "o-", label="bagging (100% attr sample)")
    plt.plot(n_trees_opts, accs_sklearn, ".", label="Sklearn RF")
    _, freqs = np.unique(y, return_counts=True)
    baseline = np.max(baseline) / y.size
    plt.hlines(y=baseline, xmin=1, xmax=300, label="Baseline", linestyles="dashdot")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test()
