"""Calculates feature importance by shuffling attributes."""
import typing as t

import numpy as np


def feat_imp(model,
             X: np.ndarray,
             y: np.ndarray,
             perf_func: t.Callable[[np.ndarray, np.ndarray], np.number],
             repetitions: int = 50,
             normalize: bool = True,
             random_state: t.Optional[int] = None) -> np.ndarray:
    """Calculates feature importance by shuffling attributes."""
    if not isinstance(repetitions, int):
        raise TypeError("'repetitions' type must be integer! "
                        "(got '{}'.)".format(type(repetitions)))

    if repetitions <= 0:
        raise ValueError("'repetitions' must be positive.")

    def get_avg_attr_imp(attr: np.ndarray,
                         base_perf: float) -> t.Tuple[float, float]:
        """Get averaged importance of current attribute."""

        cur_imp = np.zeros(repetitions)
        _attr = attr.copy()  # Keep the original form of current attr

        for i in np.arange(repetitions):
            np.random.shuffle(attr)  # Don't use random seed here
            cur_imp[i] = perf_func(y, model.predict(X))

        attr[:] = _attr  # Return 'attr' to original form
        cur_imp = base_perf - cur_imp

        return cur_imp.mean(), cur_imp.std()

    if random_state is not None:
        np.random.seed(random_state)

    imp = np.apply_along_axis(
        func1d=get_avg_attr_imp,
        arr=X,
        axis=0,
        base_perf=perf_func(y, model.predict(X)))

    if normalize:
        imp -= imp.min()
        imp /= imp.sum()

    return imp.T


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    random_seed = 1
    perf_metric = accuracy_score

    X_train, X_val, y_train, y_val = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=random_seed)

    model = RandomForestClassifier(
        n_estimators=20, random_state=random_seed).fit(X_train, y_train)

    imp = feat_imp(
        model=model,
        X=X_val,
        y=y_val,
        perf_func=perf_metric,
        random_state=random_seed)

    print("My importance: (value +/- standard deviation)\n", imp)
    print("Sklearn's importance:\n", model.feature_importances_)
