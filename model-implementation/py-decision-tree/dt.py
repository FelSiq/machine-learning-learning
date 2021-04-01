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
        assert int(depth) >= 0

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

    def set_childrens(
        self,
        child_l: t.Optional["_Node"] = None,
        child_r: t.Optional["_Node"] = None,
        set_none: bool = False,
    ) -> None:
        if set_none or child_l is not None:
            self.child_l = child_l

        if set_none or child_r is not None:
            self.child_r = child_r

    def split(
        self,
        condition: t.Any,
        features: np.ndarray,
        feat_id: int,
        impurity_fun: t.Callable[[np.ndarray], float],
        label_fun: t.Callable[[np.ndarray], t.Any],
        *args,
        **kwargs,
    ) -> t.Optional[t.Tuple["_Node", "_Node"]]:
        self.condition = condition
        self.feat_id = feat_id

        true_inds = np.array(
            [self._compare(ft, condition) for ft in features[self.inst_ids]], dtype=bool
        )

        inds_left = self.inst_ids[true_inds]
        l_labels = self.inst_labels[true_inds]

        inds_right = self.inst_ids[~true_inds]
        r_labels = self.inst_labels[~true_inds]

        l_weight = float(inds_left.size / self.inst_ids.size)

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

    def select(self, inst: np.ndarray) -> "_Node":
        assert self.child_l is not None and self.child_r is not None

        if self._compare(inst[self.feat_id], self.condition):
            return self.child_l

        return self.child_r

    def get_params(self) -> t.Dict[str, t.Any]:
        params = dict(
            inst_ids=self.inst_ids,
            inst_labels=self.inst_labels,
            impurity=self.impurity,
            label=self.label,
            depth=self.depth,
        )

        return params


class _NodeLeaf(_Node):
    def set_childrens(
        self,
        child_l: t.Optional[_Node] = None,
        child_r: t.Optional[_Node] = None,
        set_none: bool = False,
    ):
        raise NotImplementedError

    def split(self, condition: t.Any, *args, **kwargs):
        raise NotImplementedError

    def select(self, inst: np.ndarray):
        raise NotImplementedError


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


class _DecisionTreeBase:
    def __init__(
        self,
        max_depth: int = 8,
        max_node_num: int = 64,
        cat_max_comb_size: int = 2,
        min_inst_to_split: int = 2,
        min_impurity_to_split: float = 1e-2,
        num_threshold_inst_skips: int = 10,
    ):
        assert int(max_depth) >= 1
        assert int(cat_max_comb_size) >= 1
        assert int(max_node_num) >= 1
        assert int(min_inst_to_split) >= 2
        assert int(num_threshold_inst_skips) >= 1

        self.root = None
        self.max_depth = int(max_depth)
        self.max_node_num = int(max_node_num)
        self.cat_max_comb_size = int(cat_max_comb_size)
        self.min_inst_to_split = int(min_inst_to_split)
        self.min_impurity_to_split = float(min_impurity_to_split)
        self.num_threshold_inst_skips = int(num_threshold_inst_skips)

        self.col_inds_num = frozenset()
        self.col_inds_cat = frozenset()
        self._col_inds_translator = dict()

        self._node_num = 0

    def __len__(self):
        return self._node_num

    @staticmethod
    def _prepare_X_y(
        X: np.ndarray, y: np.ndarray = None
    ) -> t.Union[np.ndarray, t.Tuple[np.ndarray, np.ndarray]]:
        X = np.asarray(X)

        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        if y is not None:
            y = np.asarray(y)
            return X, y

        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        col_inds_num: t.Sequence[int],
        inst_weights: t.Optional[t.Sequence[float]] = None,
    ):
        X, y = self._prepare_X_y(X, y)
        self.dtype = y.dtype
        self.root = None

        col_inds_all = frozenset(range(X.shape[1]))
        self.col_inds_num = frozenset(col_inds_num)
        self.col_inds_cat = col_inds_all.difference(self.col_inds_num)

        sorted_numeric_vals = self._prepare_numerical(X)
        comb_by_feat = self._prepare_categorical(X)

        self._build_tree(X, y, sorted_numeric_vals, comb_by_feat, col_inds_all)

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sorted_numeric_vals: np.ndarray,
        comb_by_feat: t.Dict[int, t.Set[t.Any]],
        col_inds_all: t.FrozenSet[int],
    ):
        self._build_root(X, y, sorted_numeric_vals, comb_by_feat)

        cat_attrs_in_path = set()

        if self.root.feat_id in self.col_inds_cat:
            cat_attrs_in_path.add(self.root.feat_id)

        heap = [
            (-self.root.child_l.impurity, True, self.root, cat_attrs_in_path.copy()),
            (-self.root.child_r.impurity, False, self.root, cat_attrs_in_path.copy()),
        ]

        heapq.heapify(heap)

        while heap and self._node_num < self.max_node_num:
            _, is_left, cur_parent, cat_attrs_in_path = heapq.heappop(heap)
            cur_leaf = cur_parent.child_l if is_left else cur_parent.child_r
            cur_leaf_inst_args = cur_leaf.get_params()

            avail_feat_inds = col_inds_all.difference(cat_attrs_in_path)

            new_node = self._search_new_cut(
                X=X,
                avail_feat_ids=avail_feat_inds,
                new_node_inst_args=cur_leaf_inst_args,
                sorted_numeric_vals=sorted_numeric_vals,
                comb_by_feat=comb_by_feat,
            )

            if new_node is None:
                continue

            self._node_num += 1

            new_child_l = new_node if is_left else None
            new_child_r = None if is_left else new_node
            cur_parent.set_childrens(child_l=new_child_l, child_r=new_child_r)

            if isinstance(new_node, _NodeCategorical):
                cat_attrs_in_path = cat_attrs_in_path.copy()
                cat_attrs_in_path.add(new_node.feat_id)

            if self._can_split(new_node.child_l):
                heapq.heappush(
                    heap,
                    (-new_node.child_l.impurity, True, new_node, cat_attrs_in_path),
                )

            if self._can_split(new_node.child_r):
                heapq.heappush(
                    heap,
                    (-new_node.child_r.impurity, False, new_node, cat_attrs_in_path),
                )

    def _build_root(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sorted_numeric_vals: np.ndarray,
        comb_by_feat: t.Dict[int, t.Set[t.Any]],
    ):
        new_node_inst_args = dict(
            inst_ids=np.arange(y.size),
            inst_labels=y,
            impurity=self.impurity_fun(y),
            label=self.label_fun(y),
            depth=0,
        )

        self.root = self._search_new_cut(
            X=X,
            avail_feat_ids=np.arange(X.shape[1]),
            new_node_inst_args=new_node_inst_args,
            sorted_numeric_vals=sorted_numeric_vals,
            comb_by_feat=comb_by_feat,
        )
        self._node_num = 1

    def _can_split(self, node: _Node) -> bool:
        can_split = (
            node.depth < self.max_depth
            and node.inst_ids.size >= self.min_inst_to_split
            and node.impurity > self.min_impurity_to_split
        )
        return can_split

    def _prepare_numerical(self, X: np.ndarray) -> np.ndarray:
        if not self.col_inds_num:
            return None

        _col_inds_num_arr = np.asarray(list(self.col_inds_num), dtype=int)
        self._col_inds_translator = {cin: i for i, cin in enumerate(_col_inds_num_arr)}
        sorted_vals = np.sort(X[:, _col_inds_num_arr], axis=0)

        if self.num_threshold_inst_skips > 1:
            sorted_vals = sorted_vals[:: self.num_threshold_inst_skips, :]

        return sorted_vals

    def _prepare_categorical(self, X: np.ndarray) -> t.Dict[int, t.Set[t.Any]]:
        if not self.col_inds_cat:
            return None

        feat_uniq_vals = {
            feat_id: frozenset(X[:, feat_id]) for feat_id in self.col_inds_cat
        }

        comb_by_feat = collections.defaultdict(set)

        for feat_id, uniq_vals in feat_uniq_vals.items():
            for k in range(1, 1 + min(len(uniq_vals), self.cat_max_comb_size)):
                comb_by_feat[feat_id].update(itertools.combinations(uniq_vals, k))

        return comb_by_feat

    def _search_new_cut(
        self,
        X: np.ndarray,
        avail_feat_ids: np.ndarray,
        new_node_inst_args: t.Dict[str, t.Any],
        sorted_numeric_vals: np.ndarray,
        comb_by_feat: t.Dict[int, t.Set[t.Any]],
        chosen_node: t.Optional[_Node] = None,
    ):
        for feat_id in avail_feat_ids:
            attr = X[:, feat_id]

            if feat_id in self.col_inds_num:
                sorted_attr_ind = self._col_inds_translator[feat_id]

                chosen_node = self._create_new_cut(
                    attr,
                    feat_id,
                    new_node_inst_args,
                    True,
                    sorted_numeric_vals[:, sorted_attr_ind],
                    chosen_node=chosen_node,
                )

            else:
                chosen_node = self._create_new_cut(
                    attr,
                    feat_id,
                    new_node_inst_args,
                    False,
                    comb_by_feat[feat_id],
                    chosen_node=chosen_node,
                )

        return chosen_node

    def _create_new_cut(
        self,
        attr: np.ndarray,
        feat_id: int,
        new_node_inst_args: t.Dict[str, t.Any],
        numerical: bool,
        *args,
        chosen_node: t.Optional[_Node] = None,
    ) -> t.Optional[_Node]:
        gen_cand = functools.partial(
            _NodeNumerical if numerical else _NodeCategorical,
            **new_node_inst_args,
        )

        cand_node = gen_cand()

        if numerical:
            sorted_attr = args[0]
            conditions = 0.5 * (sorted_attr[1:] + sorted_attr[:-1])

        else:
            conditions = args[0]

        _cached_cond = set()

        for cond in conditions:

            if cond in _cached_cond:
                continue

            _cached_cond.add(cond)

            childrens = cand_node.split(
                cond,
                attr,
                feat_id,
                impurity_fun=self.impurity_fun,
                label_fun=self.label_fun,
            )

            if childrens is None:
                continue

            if (
                chosen_node is None
                or chosen_node.impurity_split > cand_node.impurity_split
            ):
                chosen_node = cand_node
                cand_node = gen_cand()

        return chosen_node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_X_y(X)
        preds = np.zeros(X.shape[0], dtype=self.dtype)

        for i, inst in enumerate(X):
            cur_node = self.root

            while not isinstance(cur_node, _NodeLeaf):
                cur_node = cur_node.select(inst)

            preds[i] = cur_node.label

        return np.squeeze(preds)


class DecisionTreeClassifier(_DecisionTreeBase):
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


class DecisionTreeRegressor(_DecisionTreeBase):
    def __init__(self, *args, **kwargs):
        super(DecisionTreeRegressor, self).__init__(*args, **kwargs)
        self.impurity_fun = np.var
        self.label_fun = np.mean


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.preprocessing
    import sklearn.tree

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X,
        y,
        shuffle=True,
        test_size=0.15,
    )

    # disc = sklearn.preprocessing.KBinsDiscretizer(encode="ordinal")
    # X_train = disc.fit_transform(X_train)
    # X_eval = disc.transform(X_eval)

    print(X.shape, y.shape)

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train, col_inds_num=list(range(X.shape[1])))
    print(len(model))
    y_preds = model.predict(X_eval)

    eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
    print(f"Eval acc: {eval_acc:.4f}")

    comparer = sklearn.tree.DecisionTreeClassifier(max_depth=3)
    comparer.fit(X_train, y_train)
    y_preds = comparer.predict(X_eval)
    eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
    print(f"Eval acc: {eval_acc:.4f}")


if __name__ == "__main__":
    _test()
