import typing as t
import heapq
import functools
import itertools
import collections

import numpy as np
import scipy.stats


class _Node:
    def __init__(
        self,
        inst_ids: np.ndarray,
        inst_labels: np.ndarray,
        impurity: float,
        label: t.Optional[t.Any] = None,
        depth: int = 0,
    ):
        assert len(inst_ids) == len(inst_labels)

        self.childrens = []  # type: t.List[_Node]
        self.inst_ids = np.asarray(inst_ids)
        self.inst_labels = np.asarray(inst_labels)
        self.feat_id = -1
        self.child_l = None  # type: t.Optional[_Node]
        self.child_r = None  # type: t.Optional[_Node]
        self.impurity = float(impurity)
        self.impurity_split = np.nan
        self.condition = None
        self.label = label
        self.depth = int(depth)

    def set_childrens(self, child_l: "_Node", child_r: "_Node"):
        self.child_l = child_l
        self.child_r = child_r

    def split(
        self,
        condition: t.Any,
        features: np.ndarray,
        feat_id: int,
        impurity_fun: t.Callable[[np.ndarray], float],
        label_fun: t.Callable[[np.ndarray], t.Any],
        *args,
        **kawrgs,
    ):
        assert len(features) == len(self.inst_ids)

        self.condition = condition
        self.feat_id = feat_id

        true_inds = np.array(
            [self._compare(ft, condition) for ft in features], dtype=bool
        )

        inds_left = self.inst_ids[true_inds]
        inds_right = self.inst_ids[~true_inds]

        l_weight = float(inds_left.size / self.inst_ids.size)

        l_labels = self.inst_labels[inds_left]
        r_labels = self.inst_labels[inds_right]

        l_impurity = impurity_fun(l_labels)
        r_impurity = impurity_fun(r_labels)

        self.impurity_split = float(
            np.dot([l_weight, 1.0 - l_weight], [l_impurity, r_impurity])
        )

        if self.impurity_split < self.impurity:
            child_l = _NodeLeaf(
                inds_left,
                l_labels,
                l_impurity,
                label=label_fun(l_labels),
                depth=self.depth + 1,
            )
            child_r = _NodeLeaf(
                inds_right,
                r_labels,
                r_impurity,
                label=label_fun(r_labels),
                depth=self.depth + 1,
            )
            self.set_childrens(child_l, child_r)
            return (inds_left, inds_right)

        return None

    def select(self, inst: np.ndarray):
        assert self.child_l is not None and self.child_r is not None

        if self._compare(inst[self.feat_id], self.condition):
            return self.child_l

        return self.child_r

    def promote(self):
        raise RuntimeError("Only leaf nodes can be promoted.")

    def demote(self):
        return _NodeLeaf(
            inst_ids=self.inst_ids,
            inst_labels=self.inst_labels,
            impurity=self.impurity,
            depth=self.depth,
        )


class _NodeLeaf(_Node):
    def set_childrens(self, child_l: _Node, child_r: _Node):
        raise NotImplementedError

    def split(self, condition: t.Any, *args, **kwargs):
        raise NotImplementedError

    def select(self, inst: np.ndarray):
        raise NotImplementedError

    def promote(self):
        return _Node(
            inst_ids=self.inst_ids,
            inst_labels=self.inst_labels,
            impurity=self.impurity,
            depth=self.depth,
        )

    def demote(self):
        raise RuntimeError("Leaf node can't be demoted.")


class _NodeNumerical(_Node):
    def __init__(self, *args, **kwargs):
        super(_NodeNumerical, self).__init__(*args, **kwargs)
        self.condition = np.nan

    @staticmethod
    def _compare(a: float, b: float):
        return a <= b

    def split(self, threshold: float, *args, **kwargs):
        assert np.isreal(threshold) and np.isfinite(threshold)
        return super(_NodeNumerical, self).split(threshold, *args, **kwargs)


class _NodeCategorical(_Node):
    def __init__(self, *args, **kwargs):
        super(_NodeCategorical, self).__init__(*args, **kwargs)
        self.condition = set()

    @staticmethod
    def _compare(a: int, b: t.FrozenSet[int]):
        return a in b

    def split(self, categories: t.Set[t.Any], *args, **kwargs):
        assert len(categories)
        categories = frozenset(categories)
        return super(_NodeCategorical, self).split(categories, *args, **kwargs)


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 8,
        max_node_num: int = 64,
        cat_max_comb_size: int = 2,
    ):
        self.root = None
        self.max_depth = int(max_depth)
        self.max_node_num = int(max_node_num)
        self.cat_max_comb_size = int(cat_max_comb_size)

        self.col_inds_num = frozenset()
        self.col_inds_cat = frozenset()

    @staticmethod
    def _prepare_X_y(X: np.ndarray, y: np.ndarray = None):
        X = np.asarray(X)

        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        if y is not None:
            y = np.asarray(y)
            return X, y

        return X

    def fit(
        self,
        X,
        y,
        col_inds_num: t.Sequence[int],
        inst_weights: t.Optional[t.Sequence[float]] = None,
    ):
        X, y = self._prepare_X_y(X, y)
        self.dtype = y.dtype
        self.root = None

        self.col_inds_num = frozenset(col_inds_num)
        self.col_inds_cat = frozenset(self.col_inds_num - set(range(X.shape[1])))

        _root_args = (y, self.impurity_fun(y), np.arange(y.size))

        if self.col_inds_num:
            sorted_numeric_vals = self._prepare_numerical(X)

        if self.col_inds_cat:
            comb_by_feat = self._prepare_categorical(X)

        for feat_id in np.arange(X.shape[1]):
            attr = X[:, feat_id]
            if feat_id in self.col_inds_num:
                self._root_cut_numerical(
                    attr, feat_id, *_root_args, sorted_numeric_vals[:, feat_id]
                )

            else:
                self._root_cut_categorical(attr, feat_id, *_root_args, comb_by_feat)

        return self

    def _prepare_numerical(self, X):
        col_inds_num_arr = np.asarray(list(self.col_inds_num), dtype=int)
        return np.sort(X[:, col_inds_num_arr], axis=0)

    def _prepare_categorical(self, X):
        feat_uniq_vals = {
            feat_id: frozenset(X[:, feat_id]) for feat_id in self.col_inds_cat
        }

        comb_by_feat = collections.defaultdict(set)

        for feat_id, uniq_vals in feat_uniq_vals.items():
            for k in range(1, 1 + min(len(uniq_vals), cat_max_comb_size)):
                comb_by_feat[feat_id].update(itertools.combinations(uniq_vals, k))

        return comb_by_feat

    @staticmethod
    def _pop_from_heap(heap):
        impurity, inst_ids, depth = heapq.heappop(heap)
        inst_labels = y[inst_ids]
        impurity *= -1
        depth += 1
        return impurity, inst_ids, inst_labels, depth

    def _root_cut_numerical(self, attr, feat_id, y, impurity, inst_ids, sorted_attr):
        gen_cand = functools.partial(
            _NodeNumerical,
            inst_ids=inst_ids,
            inst_labels=y,
            impurity=impurity,
            label=self.label_fun(y),
            depth=0,
        )

        cand_node = gen_cand()

        thresholds = 0.5 * (sorted_attr[1:] + sorted_attr[:-1])

        for threshold in thresholds:
            childrens = cand_node.split(
                threshold,
                attr,
                feat_id,
                impurity_fun=self.impurity_fun,
                label_fun=self.label_fun,
            )

            if childrens is None:
                continue

            if self.root is None or self.root.impurity_split > cand_node.impurity_split:
                self.root = cand_node
                cand_node = gen_cand()

    def _root_cut_categorical(
        self, attr, feat_id, y, impurity, inds_inds, comb_by_feat
    ):
        raise NotImplementedError

    def predict(self, X):
        X = self._prepare_X_y(X)
        preds = np.zeros(X.shape[0], dtype=self.dtype)

        for i, inst in enumerate(X):
            cur_node = self.root

            while not isinstance(cur_node, _NodeLeaf):
                cur_node = cur_node.select(inst)

            preds[i] = cur_node.label

        return np.squeeze(preds)


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, *args, **kwargs):
        super(DecisionTreeClassifier, self).__init__(*args, **kwargs)
        self.impurity_fun = self.gini
        self.label_fun = self.mode

    @staticmethod
    def mode(labels: np.ndarray):
        res = scipy.stats.mode(labels)
        return res.mode

    @staticmethod
    def gini(labels: np.ndarray):
        _, freqs = np.unique(labels, return_counts=True)
        return 1.0 - np.sum(np.square(freqs / np.sum(freqs)))


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, *args, **kwargs):
        super(DecisionTreeRegressor, self).__init__(*args, **kwargs)
        self.impurity_fun = np.var
        self.label_fun = np.mean


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics

    X, y = sklearn.datasets.load_iris(return_X_y=True)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, shuffle=True
    )

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train, col_inds_num=[0, 1, 2, 3])
    y_preds = model.predict(X_eval)

    eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
    print(f"Eval acc: {eval_acc:.4f}")


if __name__ == "__main__":
    _test()
