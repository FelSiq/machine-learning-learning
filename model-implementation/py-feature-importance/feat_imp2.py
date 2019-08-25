"""Sklearn's feature importance calculation."""
import typing as t

from sklearn.ensemble import RandomForestClassifier
import sklearn.tree
import numpy as np


def _calc_node_imp(node_id: int, tree: sklearn.tree._tree.Tree) -> float:
    """Calculate node's importance."""
    childs = np.array(
        [tree.children_left[node_id], tree.children_right[node_id]])

    node_imp = (tree.weighted_n_node_samples[node_id] * tree.impurity[node_id])

    node_imp -= (
        tree.weighted_n_node_samples[childs] * tree.impurity[childs]).sum(
            axis=0)

    return node_imp


def feat_importance(
        model: t.Union[sklearn.ensemble.forest.RandomForestClassifier, sklearn.
                       ensemble.forest.RandomForestRegressor]) -> np.ndarray:
    """Calculates feature importances like Sklearn library."""

    feat_imp = np.zeros(model.n_features_)

    for tree in model.estimators_:
        nodes_imp = np.fromfunction(
            _calc_node_imp, (tree.tree_.capacity, ),
            tree=tree.tree_,
            dtype=int)

        cur_feat_imp = np.array([
            nodes_imp[tree.tree_.feature == i].sum()
            for i in np.arange(model.n_features_)
        ]) / nodes_imp.sum()

        feat_imp += cur_feat_imp / cur_feat_imp.sum()

    return feat_imp / model.n_estimators


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100).fit(
        iris.data, iris.target)

    imp = feat_importance(model)

    print("Feature importance:", imp)
    print("Sklearn feat importance:", model.feature_importances_)

    assert np.allclose(imp, model.feature_importances_)
