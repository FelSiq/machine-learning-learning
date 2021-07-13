import numpy as np
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.svm
import sklearn.ensemble
import sklearn.datasets
import sklearn.cluster

"""
tl,dr; if using l1 reg, threshold is default to 1e-5. Otherwise, threshold is the mean value of scores.

from sklearn.feature_selection.SelectFromModel documentation:

https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel

thresholds : tring or float, default=None
    The threshold value to use for feature selection. Features whose importance is greater or equal are kept while the others are discarded. If “median” (resp. “mean”), then the threshold value is the median (resp. the mean) of the feature importances. A scaling factor (e.g., “1.25*mean”) may also be used. If None and if the estimator has a parameter penalty set to l1, either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5. Otherwise, “mean” is used by default.

"""


X, y = sklearn.datasets.load_wine(return_X_y=True)

# 1) Feature selection with L1 regularization
# 1.a) With a simple Lasso regression
# Higher alpha -> less features
model = sklearn.linear_model.Lasso(alpha=0.25)
model.fit(X, y)
selector = sklearn.feature_selection.SelectFromModel(model, prefit=True)
X_sel = selector.transform(X)
print(
    f"(Lasso Reg) Lasso regression dimension reduction: {1 - X_sel.shape[1] / X.shape[1]:.3f}"
)

# 1.b) With a LinearSVM
# Smaller C -> less features
model = sklearn.svm.LinearSVC(C=0.005, penalty="l1", dual=False)
model.fit(X, y)
selector = sklearn.feature_selection.SelectFromModel(model, prefit=True)
X_sel = selector.transform(X)
print(
    f"(LinearSVC) Lasso regression dimension reduction: {1 - X_sel.shape[1] / X.shape[1]:.3f}"
)


# 2) Feature selection with Trees/Forests
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
model.fit(X, y)
selector = sklearn.feature_selection.SelectFromModel(model, prefit=True)
X_sel = selector.transform(X)
print(f"RF dimension reduction: {1 - X_sel.shape[1] / X.shape[1]:.3f}")


# 3) Feature selection with iterative forward/backward selection
# 3.a) backwards
estimator = sklearn.linear_model.LinearRegression()
selector = sklearn.feature_selection.SequentialFeatureSelector(
    estimator, n_features_to_select=0.7, direction="backward", cv=5
)
selector.fit(X, y)
selected_feats = selector.get_support()
X_sel = selector.transform(X)
print(f"backward feat selection dim red: {1 - X_sel.shape[1] / X.shape[1]:.3f}")

# 3.a) forward
estimator = sklearn.linear_model.LinearRegression()
selector = sklearn.feature_selection.SequentialFeatureSelector(
    estimator, n_features_to_select=0.7, direction="forward", cv=5
)
selector.fit(X, y)
selected_feats = selector.get_support()
X_sel = selector.transform(X)
print(f"forward feat selection dim red: {1 - X_sel.shape[1] / X.shape[1]:.3f}")


# 4) Maybe not really a feature selection, but a dim reduction technique:
# use AgglomerativeClustering to cluster features instead of instances.
selector = sklearn.cluster.FeatureAgglomeration(n_clusters=4)
selector.fit(X, y)
X_sel = selector.transform(X)
print(f"Agglomerative clustering dim red: {1 - X_sel.shape[1] / X.shape[1]:.3f}")
