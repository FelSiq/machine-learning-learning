"""Decision tree implementation."""
import typing as t

import numpy as np


class DecisionTreeClassifier:
    """Generic Decision Tree Classifier model."""

    def __init__(self,
                 max_depth: t.Optional[int] = None,
                 min_impurity: float = 0.0):
        """Init a Decision Tree Classifier model."""
        self.max_depth = np.inf if max_depth is None else max_depth
        self.min_impurity = min_impurity

    @staticmethod
    def _impurity(collection: np.ndarray) -> float:
        """Measures impurity of a collection."""
        _, freqs = np.unique(collection, return_counts=True)
        return 1.0 - np.sum(np.square(freqs / collection.size))

    @staticmethod
    def _get_node_class(y: np.ndarray) -> t.Any:
        """Get the node class given by the majority vote."""
        classes, freqs = np.unique(y, return_counts=True)
        return classes[np.argmax(freqs)]

    @classmethod
    def gini_impurity(
            cls,
            class_l: np.ndarray,
            class_r: np.ndarray,
            return_classes_imp: bool = False,
    ) -> t.Union[float, t.Tuple[float, float, float]]:
        """Weighted Gini impurity for a binary split."""

        total_size = class_l.size + class_r.size

        class_imp_l = cls._impurity(class_l)
        class_imp_r = cls._impurity(class_r)

        gini_impurity = (class_l.size / total_size * class_imp_l +
                         class_r.size / total_size * class_imp_r)

        if return_classes_imp:
            return gini_impurity, class_imp_l, class_imp_r

        return gini_impurity

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            attr_sample_frac: float = 1.0,
            random_state: t.Optional[int] = None) -> "DecisionTreeClassifier":
        """Method to be overwritten by the subclasses."""
        # pylint: disable=W0613
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Method to be overwritten by the subclasses."""
        # pylint: disable=W0613, R0201
        return np.array([])


class DecisionTreeNumeric(DecisionTreeClassifier):
    """Decision Tree Classifier model for numeric data only."""

    def __init__(self,
                 max_depth: t.Optional[int] = None,
                 min_impurity: float = 0.0):
        """Init a Decision Tree Classifier model for numeric data."""
        super().__init__(max_depth=max_depth, min_impurity=min_impurity)

        self._child_l = np.array(
            [], dtype=np.uint16)  # type: t.Union[np.ndarray, t.List[int]]
        self._child_r = np.array(
            [], dtype=np.uint16)  # type: t.Union[np.ndarray, t.List[int]]
        self._node_cls = np.array(
            [], dtype=object)  # type: t.Union[np.ndarray, t.List[t.Any]]
        self._node_threshold = np.array(
            [], dtype=float)  # type: t.Union[np.ndarray, t.List[float]]
        self._node_impurity = np.array(
            [], dtype=float)  # type: t.Union[np.ndarray, t.List[float]]
        self._node_depth = np.array(
            [], dtype=int)  # type: t.Union[np.ndarray, t.List[int]]
        self._node_attr = np.array(
            [], dtype=int)  # type: t.Union[np.ndarray, t.List[int]]

    def _create_nodes(self, node_num: int) -> None:
        """Create a new node on the tree model."""
        self._node_cls += node_num * [None]
        self._child_l += node_num * [-1]
        self._child_r += node_num * [-1]
        self._node_threshold += node_num * [np.inf]
        self._node_depth += node_num * [0]
        self._node_attr += node_num * [-1]
        self._node_impurity += node_num * [np.inf]

    def _create_split(
            self, X: np.ndarray, y: np.ndarray
    ) -> t.Tuple[int, float, np.ndarray, np.ndarray, float, float]:
        """Get the current best binary split on the given data."""
        best_impurity = np.inf
        best_imp_l = np.inf
        best_imp_r = np.inf
        threshold = np.nan
        used_attr_ind = np.nan
        inds_inst_l = None
        inds_inst_r = None

        for attr_ind in np.arange(X.shape[1]):
            sorted_inds = np.argsort(X[:, attr_ind])
            X_sorted = X[sorted_inds, :]
            y_sorted = y[sorted_inds]

            for cut_ind in np.arange(1, sorted_inds.size):
                class_l_inds = sorted_inds[:cut_ind]
                class_r_inds = sorted_inds[cut_ind:]

                cur_impurity, imp_l, imp_r = self.gini_impurity(  # type: ignore
                    class_l=y_sorted[class_l_inds],
                    class_r=y_sorted[class_r_inds],
                    return_classes_imp=True)

                if cur_impurity < best_impurity:
                    best_impurity = cur_impurity
                    threshold = 0.5 * (X_sorted[cut_ind - 1, attr_ind] +
                                       X_sorted[cut_ind, attr_ind])
                    used_attr_ind = attr_ind
                    inds_inst_l = np.copy(class_l_inds)
                    inds_inst_r = np.copy(class_r_inds)
                    best_imp_l = imp_l
                    best_imp_r = imp_r

        return (used_attr_ind, threshold, inds_inst_l, inds_inst_r, best_imp_l,
                best_imp_r)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            attr_sample_frac: float = 1.0,
            random_state: t.Optional[int] = None) -> "DecisionTreeClassifier":
        """Build a Decision Tree Classifier model using given training data."""
        if not 0 < attr_sample_frac <= 1.0:
            raise ValueError("'attr_sample_frac' must be in (0, 1] (got {}.)".
                             format(attr_sample_frac))

        if random_state is not None:
            np.random.seed(random_state)

        X = np.copy(X)
        y = np.copy(y)

        self._child_l = []
        self._child_r = []
        self._node_cls = []
        self._node_threshold = []
        self._node_impurity = []
        self._node_depth = []
        self._node_attr = []

        self._create_nodes(node_num=1)

        self._node_impurity = [self._impurity(y)]

        split_stack = [(0, np.arange(y.size), np.array([], dtype=int))
                       ]  # type: t.List[t.Tuple[int, np.ndarray, t.List[int]]]
        next_ind = 0

        while split_stack:
            cur_node_ind, inst_inds_node, used_attrs = split_stack.pop()

            self._node_cls[cur_node_ind] = self._get_node_class(
                y=y[inst_inds_node])
            available_attrs = np.delete(np.arange(X.shape[1]), used_attrs)

            if (self._node_impurity[cur_node_ind] <= self.min_impurity
                    or self._node_depth[cur_node_ind] >= self.max_depth
                    or available_attrs.size == 0):
                continue

            attr_num_split = max(
                1, round(attr_sample_frac * available_attrs.size))
            attrs_sample = np.random.choice(
                a=available_attrs, size=attr_num_split, replace=False)

            X_sample = X[inst_inds_node, :][:, attrs_sample]

            used_attr_ind, threshold, inds_inst_l, inds_inst_r, imp_l, imp_r = (
                self._create_split(X=X_sample, y=y))

            used_attrs = np.hstack((used_attrs, attrs_sample[used_attr_ind]))
            split_stack.append((next_ind + 1, inst_inds_node[inds_inst_l],
                                used_attrs))
            split_stack.append((next_ind + 2, inst_inds_node[inds_inst_r],
                                used_attrs))

            self._create_nodes(node_num=2)
            self._child_l[cur_node_ind] = next_ind + 1
            self._child_r[cur_node_ind] = next_ind + 2
            self._node_impurity[next_ind + 1] = imp_l
            self._node_impurity[next_ind + 2] = imp_r
            self._node_depth[next_ind + 1] = self._node_depth[cur_node_ind] + 1
            self._node_depth[next_ind + 2] = self._node_depth[cur_node_ind] + 1
            self._node_threshold[cur_node_ind] = threshold
            self._node_attr[cur_node_ind] = attrs_sample[used_attr_ind]

            next_ind += 2

        self._child_l = np.asarray(self._child_l, dtype=int)
        self._child_r = np.asarray(self._child_r, dtype=int)
        self._node_cls = np.asarray(self._node_cls, dtype=object)
        self._node_threshold = np.asarray(self._node_threshold, dtype=float)
        self._node_impurity = np.asarray(self._node_impurity, dtype=float)
        self._node_depth = np.asarray(self._node_depth, dtype=int)
        self._node_attr = np.asarray(self._node_attr, dtype=int)

        return self

    def _node_is_leaf(self, cur_node_ind: int) -> bool:
        """Check if a given node is a leaf node."""
        return (self._child_l[cur_node_ind] == -1
                and self._child_r[cur_node_ind] == -1)

    def get_leaves_impurity(self) -> np.ndarray:
        """Return a list of impurity for all leaf nodes."""
        impurity = []

        for node_ind in np.arange(len(self._node_impurity)):
            if self._node_is_leaf(node_ind):
                impurity.append(self._node_impurity[node_ind])

        return np.asarray(impurity)

    def _predict_inst(self, inst: np.ndarray) -> t.Any:
        """Predict the class of a single instance."""
        cur_node_ind = 0

        while not self._node_is_leaf(cur_node_ind):
            if inst[self._node_attr[cur_node_ind]] <= self._node_threshold[
                    cur_node_ind]:
                cur_node_ind = self._child_l[cur_node_ind]

            else:
                cur_node_ind = self._child_r[cur_node_ind]

        return self._node_cls[cur_node_ind]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of each ``X`` instance."""
        return np.apply_along_axis(arr=X, axis=1, func1d=self._predict_inst)


class DecisionTreeCategorical(DecisionTreeClassifier):
    """Decision Tree Classifier model for categorical data."""

    def __init__(self,
                 max_depth: t.Optional[int] = None,
                 min_impurity: float = 0.0):
        """Init a Decision Tree Classifier model for categorical data."""
        super().__init__(max_depth=max_depth, min_impurity=min_impurity)

        raise NotImplementedError("Not implemented.")


class DecisionTreeMixed(DecisionTreeClassifier):
    """Decision Tree Classifier model for mixed data."""

    def __init__(self,
                 max_depth: t.Optional[int] = None,
                 min_impurity: float = 0.0):
        """Init a Decision Tree Classifier model for mixed data."""
        super().__init__(max_depth=max_depth, min_impurity=min_impurity)

        raise NotImplementedError("Not implemented.")


def _test_01() -> None:
    # pylint: disable=E1101, C0415
    from sklearn.datasets import load_iris

    iris = load_iris()
    model = DecisionTreeNumeric()
    model.fit(iris.data, iris.target)
    print("Accuracy:", np.mean(model.predict(iris.data) == iris.target))
    print("Leaves impurity:", model.get_leaves_impurity())


def _test_02() -> None:
    # pylint: disable=E1101, C0415
    from sklearn.datasets import load_iris

    iris = load_iris()
    model = DecisionTreeNumeric()
    model.fit(iris.data, iris.target, attr_sample_frac=0.5)
    print("Accuracy:", np.mean(model.predict(iris.data) == iris.target))
    print("Leaves impurity:", model.get_leaves_impurity())


def _test_03() -> None:
    # pylint: disable=E1101, C0415
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier as SklearnDT

    iris = load_iris()
    n_splits = 10
    splitter = StratifiedKFold(n_splits=n_splits)

    model = DecisionTreeNumeric()
    model_skl = SklearnDT()

    acc = 0
    acc_skl = 0
    for inds_train, inds_test in splitter.split(iris.data, iris.target):
        model.fit(iris.data[inds_train, :], iris.target[inds_train])
        model_skl.fit(iris.data[inds_train, :], iris.target[inds_train])

        acc += np.mean(
            model.predict(iris.data[inds_test, :]) == iris.target[inds_test])
        acc_skl += np.mean(
            model_skl.predict(iris.data[inds_test, :]) == iris.
            target[inds_test])

    print("Estimated accuracy: {:.4f}".format(acc / n_splits))
    print("Estimated sklearn accuracy: {:.4f}".format(acc_skl / n_splits))


if __name__ == "__main__":
    _test_01()
    _test_02()
    _test_03()
