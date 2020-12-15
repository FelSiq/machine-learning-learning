"""Promote a exploratory search on metadataset
focusing attention to the predictive attributes.
The purpose of this script is to identify which
metafeatures are the most important in the meta-
dataset producet by MFE.
"""
# https://scikit-learn.org/stable/modules/feature_selection.html
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np


# Script configuration
METAFEATURES_PATH = "./metafeatures/"
METAFEATURES_FILENAME = "final_combined-accuracy.metadata"
PERF_COLUMNS_INDEX = [-4, -3, -2, -1]
IGNORE_COLUMNS_INDEX = [0, 1]
RETAINED_COLUMNS_PROP = 0.25
CV_BINS = 20
ENABLE_SELECT_KBEST = True
ENABLE_RFECV = True
ENABLE_SELECT_FROM_MODEL = True
EMSEMBLE_TREE_NUM = 100

# Load metadataset
metafeatures = pd.read_csv(
        METAFEATURES_PATH +
        METAFEATURES_FILENAME)
metafeatures.fillna(0, inplace=True)

X = metafeatures.drop(metafeatures.columns[
    IGNORE_COLUMNS_INDEX + PERF_COLUMNS_INDEX], axis=1)
y = metafeatures.iloc[:, PERF_COLUMNS_INDEX].values.astype(np.float64)
base_learner_name = metafeatures.columns[PERF_COLUMNS_INDEX]


# Select K Best method
def select_k_best_report(X, y, metric):
    """."""
    relevant_features = np.zeros(X.shape[1])
    retained_columns = int(np.ceil(X.shape[1] * RETAINED_COLUMNS_PROP))

    k_selecter = SelectKBest(
        metric,
        k=retained_columns)

    for classifier_index in range(y.shape[1]):
        k_selecter.fit_transform(
                        X,
                        y[:, classifier_index])
        selected_features = k_selecter.get_support()
        relevant_features[selected_features] += 1

    ans = {
        "total_columns_originally": X.shape[1],
        "retained_columns_by_iteration": retained_columns,
        "relevant_feat_cardinality": sum(relevant_features > 0),
        "reduction": sum(relevant_features > 0) / X.shape[1],
        "relevant_features": X.columns[relevant_features > 0],
    }

    return ans


# Analysis of Select K Best method results
if ENABLE_SELECT_KBEST:
    print("---- SELECT K BEST ----")
    ans_f = select_k_best_report(X, y, f_regression)
    ans_m = select_k_best_report(X, y, mutual_info_regression)
    ans_combined = pd.DataFrame(
            [ans_f, ans_m],
            index=["f_reg", "mutual_info_reg"]).transpose()

    print(ans_combined.drop("relevant_features"))

    common_relevant_feat = set(ans_f["relevant_features"]).\
        intersection(ans_m["relevant_features"])

    print("Intersection of relevant feats.:", common_relevant_feat)
    print("Cardinality of common relevant feats:", len(common_relevant_feat))
    print()


# Recursive Feature Elimination (RFE) with Cross Validation (CV)
def rfe_test(X, y, cv=10, metric="neg_mean_squared_error"):
    recommender = DecisionTreeRegressor()

    rfecv = RFECV(
        estimator=recommender,
        cv=CV_BINS,
        scoring=metric)

    ans = {}
    features = {}
    for i in range(y.shape[1]):
        rfecv.fit(X, y[:, i])
        features_selected = X.columns[rfecv.get_support()]

        ans[base_learner_name[i]] = {
            "number_of_feat": rfecv.n_features_,
            "proportion": rfecv.n_features_ / X.shape[1],
        }
        features[base_learner_name[i]] = features_selected

    ans = pd.DataFrame(ans).transpose()
    ans["number_of_feat"] = ans["number_of_feat"].astype(np.int32)

    return ans, features


if ENABLE_RFECV:
    print("---- RFECV ----")
    ans_rfe, ft_rfe = rfe_test(
        X,
        y,
        cv=CV_BINS,
        metric="explained_variance")

    # Analysis of RFECV results
    print(ans_rfe)
    print(ft_rfe)
    print()


# Select from model
def select_from_model_test(X, y, recommender):
    ans = {}
    for i in range(y.shape[1]):
        recommender.fit(X, y[:, i])
        selecter = SelectFromModel(recommender, prefit=True)
        selected_feats = X.columns[selecter.get_support()]
        ans[base_learner_name[i]] = selected_feats

    return ans


if ENABLE_SELECT_FROM_MODEL:
    print("---- SELECT FROM MODEL ----")
    recommender_lasso = Lasso(max_iter=1e+5)
    recommender_emstree = ExtraTreesClassifier(n_estimators=EMSEMBLE_TREE_NUM)

    recommender_list = [
        recommender_lasso,
        recommender_emstree
    ]

    for recommender in recommender_list:
        print("Current recommender:", recommender.__class__)
        ans_sfm = select_from_model_test(X, y, recommender)
        unanimity_set = set(X.columns)
        for base_learner in ans_sfm:
            cur_features = ans_sfm[base_learner]
            # print(base_learner, ":", cur_features)
            unanimity_set.intersection_update(cur_features)

        print("Unanimity features:", unanimity_set, end="\n\n")
    print()
