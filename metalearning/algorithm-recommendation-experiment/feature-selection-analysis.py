"""Promote a exploratory search on metadataset
focusing attention to the predictive attributes.
The purpose of this script is to identify which
metafeatures are the most important in the meta-
dataset producet by MFE.
"""
# https://scikit-learn.org/stable/modules/feature_selection.html
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.feature_selection import f_regression, mutual_info_regression
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


# Script configuration
METAFEATURES_PATH = "./metafeatures/"
METAFEATURES_FILENAME = "final_combined-accuracy.metadata"
PERF_COLUMNS_INDEX = [-4, -3, -2, -1]
IGNORE_COLUMNS_INDEX = [0, 1]
RETAINED_COLUMNS_PROP = 0.25

# Load metadataset
metafeatures = pd.read_csv(
        METAFEATURES_PATH +
        METAFEATURES_FILENAME)
metafeatures.fillna(0, inplace=True)

X = metafeatures.drop(metafeatures.columns[
    IGNORE_COLUMNS_INDEX + PERF_COLUMNS_INDEX], axis=1)
y = metafeatures.iloc[:, PERF_COLUMNS_INDEX].values.astype(np.float64)


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

# Recursive Feature Elimination (RFE) with Cross Validation (CV)
