import typing as t
import heapq
import functools
import itertools
import collections
import multiprocessing
import bisect

import numpy as np
import sklearn.utils.extmath


class _Node:
    def __init__(
        self,
        inst_ids: np.ndarray,
        inst_labels: np.ndarray,
        inst_weight: np.ndarray,
        impurity: float,
        label: t.Optional[t.Any] = None,
        depth: int = 0,
    ):
        assert len(inst_ids) == len(inst_labels)
        assert int(depth) >= 0

        self.childrens = []  # type: t.List[_Node]
        self.inst_ids = np.asarray(inst_ids, dtype=int)
        self.inst_labels = np.asarray(inst_labels)
        self.inst_weight = np.asfarray(inst_weight)
        self.feat_id = -1
        self.child_l = None  # type: t.Optional[_Node]
        self.child_r = None  # type: t.Optional[_Node]
        self.impurity = float(impurity)
        self.impurity_split = np.inf
        self.condition = None
        self.label = label
        self.depth = int(depth)
        self.l_child_weight = np.nan
        self.r_child_weight = np.nan

    def __lt__(self, other: "_Node") -> bool:
        return True

    def __gt__(self, other: "_Node") -> bool:
        return True

    def __eq__(self, other: "_Node") -> bool:
        return True

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
        self.impurity_split = np.inf

        true_inds = np.array(
            [self._compare(ft, condition) for ft in features[self.inst_ids]], dtype=bool
        )

        inds_left = self.inst_ids[true_inds]
        inds_right = self.inst_ids[~true_inds]

        if inds_left.size == 0 or inds_right.size == 0:
            return None

        l_labels = self.inst_labels[true_inds]
        r_labels = self.inst_labels[~true_inds]

        l_weight = self.inst_weight[true_inds]
        r_weight = self.inst_weight[~true_inds]

        l_child_weight = float(np.sum(l_weight)) / float(np.sum(self.inst_weight))
        r_child_weight = 1.0 - l_child_weight

        l_impurity = impurity_fun(l_labels, l_weight)
        r_impurity = impurity_fun(r_labels, r_weight)

        impurity_split = float(
            np.dot([l_child_weight, r_child_weight], [l_impurity, r_impurity])
        )

        if impurity_split < self.impurity:
            child_l = _NodeLeaf(
                inds_left,
                l_labels,
                l_weight,
                l_impurity,
                label=label_fun(l_labels, l_weight),
                depth=self.depth + 1,
            )
            child_r = _NodeLeaf(
                inds_right,
                r_labels,
                r_weight,
                r_impurity,
                label=label_fun(r_labels, r_weight),
                depth=self.depth + 1,
            )

            self.set_childrens(child_l, child_r)

            self.condition = condition
            self.feat_id = feat_id
            self.l_child_weight = l_child_weight
            self.r_child_weight = r_child_weight
            self.impurity_split = impurity_split

            return (inds_left, inds_right)

        self.l_child_weight = np.nan
        self.r_child_weight = np.nan
        self.impurity_split = np.nan

        return None

    def select(self, inst: np.ndarray) -> "_Node":
        assert self.child_l is not None and self.child_r is not None
        assert inst.ndim == 1

        if self._compare(inst[self.feat_id], self.condition):
            return self.child_l

        return self.child_r

    def get_params(self) -> t.Dict[str, t.Any]:
        params = dict(
            inst_ids=self.inst_ids,
            inst_labels=self.inst_labels,
            inst_weight=self.inst_weight,
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
        max_depth: int = 32,
        max_node_num: int = 64,
        cat_max_comb_size: int = 2,
        min_inst_to_split: int = 2,
        min_impurity_to_split: float = 0.0,
        num_threshold_inst_skips: t.Union[float, int] = 0.05,
        num_workers: int = -1,
    ):
        assert int(max_depth) >= 1
        assert int(cat_max_comb_size) >= 1
        assert int(max_node_num) >= 1
        assert int(min_inst_to_split) >= 2
        assert num_threshold_inst_skips > 0

        self.root = None
        self.max_depth = int(max_depth)
        self.max_node_num = int(max_node_num)
        self.cat_max_comb_size = int(cat_max_comb_size)
        self.min_inst_to_split = int(min_inst_to_split)
        self.min_impurity_to_split = float(min_impurity_to_split)
        self.num_threshold_inst_skips = num_threshold_inst_skips
        self.num_workers = (
            int(num_workers) if num_workers >= 1 else multiprocessing.cpu_count()
        )

        self.col_inds_num = frozenset()
        self.col_inds_cat = frozenset()
        self._col_inds_translator = dict()

        self._node_num = 0
        self.dtype = None

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
        sample_weight: t.Optional[t.Sequence[float]] = None,
        col_inds_num: t.Sequence[int] = None,
    ):
        X, y = self._prepare_X_y(X, y)
        self.root = None

        if self.dtype is None:
            self.dtype = y.dtype

        if sample_weight is None:
            sample_weight = np.full(y.size, fill_value=1.0 / y.size)

        if col_inds_num is None:
            col_inds_num = list(range(X.shape[1]))

        col_inds_all = frozenset(range(X.shape[1]))
        self.col_inds_num = frozenset(col_inds_num)
        self.col_inds_cat = col_inds_all.difference(self.col_inds_num)

        sorted_numeric_vals = self._prepare_numerical(X)
        comb_by_feat = self._prepare_categorical(X)

        self._build_tree(
            X, y, sample_weight, sorted_numeric_vals, comb_by_feat, col_inds_all
        )

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        sorted_numeric_vals: np.ndarray,
        comb_by_feat: t.Dict[int, t.Set[t.Any]],
        col_inds_all: t.FrozenSet[int],
    ):
        self._build_root(
            X, y, sample_weight, sorted_numeric_vals, comb_by_feat, col_inds_all
        )

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
        sample_weight: np.ndarray,
        sorted_numeric_vals: np.ndarray,
        comb_by_feat: t.Dict[int, t.Set[t.Any]],
        col_inds_all: t.FrozenSet[int],
    ):
        new_node_inst_args = dict(
            inst_ids=np.arange(y.size),
            inst_labels=y,
            inst_weight=sample_weight,
            impurity=self.impurity_fun(y, sample_weight),
            label=self.label_fun(y, sample_weight),
            depth=0,
        )

        self.root = self._search_new_cut(
            X=X,
            avail_feat_ids=col_inds_all,
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

        if self.num_threshold_inst_skips != 1:
            skips = self.num_threshold_inst_skips

            if 0 < skips < 1:
                skips = int(np.ceil(X.shape[0] * skips))

            sorted_vals = sorted_vals[::skips, :]

            if sorted_vals.shape[0] % skips:
                sorted_vals = np.vstack((sorted_vals, sorted_vals[-1, :]))

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
            min_, max_ = np.quantile(attr[new_node_inst_args["inst_ids"]], (0, 1))
            start = bisect.bisect_left(sorted_attr, min_)
            end = bisect.bisect_right(sorted_attr, max_)
            sorted_attr = sorted_attr[start:end]
            conditions = 0.5 * (sorted_attr[:-1] + sorted_attr[1:])

        else:
            conditions = args[0]

        _last_cond = -np.inf

        for cond in conditions:
            if (numerical and np.isclose(cond, _last_cond)) or (cond == _last_cond):
                continue

            _last_cond = cond

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

    def _pred_instance(self, inst: np.ndarray):
        cur_node = self.root

        while not isinstance(cur_node, _NodeLeaf):
            cur_node = cur_node.select(inst)

        return cur_node.label

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_X_y(X)

        with multiprocessing.Pool(self.num_workers) as pool:
            preds = pool.map(self._pred_instance, X)

        preds = np.hstack(preds).astype(dtype=self.dtype, copy=False)
        return np.squeeze(preds)


class DecisionTreeClassifier(_DecisionTreeBase):
    def __init__(self, *args, **kwargs):
        super(DecisionTreeClassifier, self).__init__(*args, **kwargs)
        self.impurity_fun = self.weighted_gini
        self.label_fun = self.weighted_mode
        self.dtype = None

    @staticmethod
    def weighted_mode(labels: np.ndarray, sample_weight: np.ndarray) -> t.Any:
        cls, _ = sklearn.utils.extmath.weighted_mode(a=labels, w=sample_weight)
        return np.squeeze(cls)

    @staticmethod
    def weighted_gini(labels: np.ndarray, sample_weight: np.ndarray) -> float:
        _, inv_inds, freqs = np.unique(labels, return_inverse=True, return_counts=True)

        cls_weights = np.zeros(freqs.size, dtype=float)
        np.add.at(cls_weights, inv_inds, sample_weight)

        weighted_freqs = freqs * cls_weights
        total_cls_weights = float(np.sum(weighted_freqs))

        w_gini = 1.0 - float(np.sum(np.square(weighted_freqs / total_cls_weights)))

        return w_gini


class DecisionTreeRegressor(_DecisionTreeBase):
    def __init__(self, *args, **kwargs):
        super(DecisionTreeRegressor, self).__init__(*args, **kwargs)
        self.impurity_fun = self.weighted_var
        self.label_fun = self.weighted_avg
        self.dtype = float

    @staticmethod
    def weighted_avg(labels: np.ndarray, sample_weight: np.ndarray) -> float:
        return float(np.average(labels, weights=sample_weight))

    @staticmethod
    def weighted_var(
        labels: np.ndarray, sample_weight: np.ndarray, bias_correction: bool = True
    ) -> float:
        if labels.size == 1:
            return 0.0

        weighted_avg = np.average(labels, weights=sample_weight)

        weighted_var = np.dot(sample_weight, np.square(labels - weighted_avg))
        weighted_var /= float(np.sum(sample_weight))

        if bias_correction:
            sqr_sum_sample_weight = np.square(np.sum(sample_weight))
            sum_sqr_sample_weight = np.sum(np.square(sample_weight))

            weighted_var *= sqr_sum_sample_weight / (
                sqr_sum_sample_weight - sum_sqr_sample_weight
            )

        return float(weighted_var)


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.preprocessing
    import sklearn.tree

    X, y = sklearn.datasets.load_wine(return_X_y=True)

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

    args_a = dict(max_depth=80, max_node_num=256, min_inst_to_split=2)
    args_b = dict(max_depth=80, min_samples_split=2)

    model = DecisionTreeClassifier(**args_a)

    sample_weight = None

    if isinstance(model, DecisionTreeClassifier):
        _, freqs = np.unique(y_train, return_counts=True)
        sample_weight = freqs[y_train].astype(float)
        sample_weight /= float(np.sum(sample_weight))

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )
    print(len(model))
    y_preds = model.predict(X_eval)

    if isinstance(model, DecisionTreeClassifier):
        eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
        print(f"Eval acc: {eval_acc:.4f}")
        comparer = sklearn.tree.DecisionTreeClassifier(**args_b)

    else:
        eval_rmse = sklearn.metrics.mean_squared_error(y_preds, y_eval, squared=False)
        print(f"Eval rmse: {eval_rmse:.4f}")
        comparer = sklearn.tree.DecisionTreeRegressor(**args_b)

    comparer.fit(X_train, y_train, sample_weight=sample_weight)
    y_preds = comparer.predict(X_eval)

    if isinstance(model, DecisionTreeClassifier):
        eval_acc = sklearn.metrics.accuracy_score(y_preds, y_eval)
        print(f"Eval acc: {eval_acc:.4f}")

    else:
        eval_rmse = sklearn.metrics.mean_squared_error(y_preds, y_eval, squared=False)
        print(f"Eval rmse: {eval_rmse:.4f}")


if __name__ == "__main__":
    _test()
