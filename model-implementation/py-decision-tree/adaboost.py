# Implementing AdaBoost using SAMME variance
# Source: https://web.stanford.edu/~hastie/Papers/samme.pdf
import typing as t

import numpy as np
import tqdm.auto


class AdaBoost:
    def __init__(
        self,
        weak_learner_gen,
        perf_metric: t.Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        max_learners: int = 64,
    ):
        assert max_learners > 1

        self.weak_learner_gen = weak_learner_gen
        self.perf_metric = perf_metric

        self.max_learners = int(max_learners)

        self.ensemble = []

        self._classes = np.empty(0)

    def __len__(self):
        return len(self.ensemble)

    def _calc_weak_learner_weight(
        self,
        y_preds: np.ndarray,
        y_true: np.ndarray,
        inst_weights: np.ndarray,
        eps: float = 1e-7,
    ):
        incorrect = y_preds != y_true
        total_err = eps + float(np.sum(inst_weights[incorrect]))
        total_err /= float(np.sum(inst_weights))

        n = self._classes.size

        learner_weight = float(np.log(n - 1) + np.log((1.0 - total_err) / total_err))

        return total_err, learner_weight

    @staticmethod
    def _update_class_weights(
        y_preds: np.ndarray,
        y_true: np.ndarray,
        inst_weights: np.ndarray,
        weak_learner_weight: float,
    ):
        incorrect = y_preds != y_true
        new_weights = inst_weights * np.exp(
            incorrect * weak_learner_weight * (inst_weights > 0)
        )
        new_weights /= float(np.sum(new_weights))

        return new_weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._classes = np.unique(y)

        inst_weights = np.full(y.size, fill_value=1.0 / y.size)

        self.ensemble = []

        for _ in tqdm.auto.tqdm(np.arange(self.max_learners), leave=False):
            weak_learner = self.weak_learner_gen()
            weak_learner.fit(X, y, sample_weight=inst_weights)
            y_preds = weak_learner.predict(X)

            if np.isclose(np.mean(y_preds == y), 1.0):
                break

            learner_total_err, weak_learner_weight = self._calc_weak_learner_weight(
                y_preds, y, inst_weights
            )

            if abs(weak_learner_weight) < 1e-6:
                break

            inst_weights = self._update_class_weights(
                y_preds, y, inst_weights, weak_learner_weight
            )

            self.ensemble.append((weak_learner, weak_learner_weight))

        return self

    def predict(self, X):
        votes = np.zeros((self._classes.size, X.shape[0]), dtype=float)

        for model, model_weight in self.ensemble:
            y_preds = model.predict(X)
            votes[y_preds, np.arange(X.shape[0])] += model_weight

        return np.argmax(votes, axis=0)


def _test():
    import sklearn.tree
    import sklearn.model_selection
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.ensemble
    import functools
    import dt

    def weighted_acc(y_pred, y_true, weight):
        return sklearn.metrics.accuracy_score(y_pred, y_true, sample_weight=weight)

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    perf_metric = weighted_acc

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, shuffle=True
    )

    for wlg in [sklearn.tree.DecisionTreeClassifier, dt.DecisionTreeClassifier]:
        weak_learner_gen = functools.partial(wlg, max_depth=1)

        booster = AdaBoost(weak_learner_gen, perf_metric=perf_metric, max_learners=20)
        booster.fit(X_train, y_train)
        print(len(booster))

        y_preds = booster.predict(X_eval)

        eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
        print(f"Eval acc: {eval_acc:.4f}")

    comparer = sklearn.ensemble.AdaBoostClassifier(algorithm="SAMME")
    comparer.fit(X_train, y_train)
    y_preds = comparer.predict(X_eval)
    eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
    print(f"Eval acc: {eval_acc:.4f}")


if __name__ == "__main__":
    _test()
