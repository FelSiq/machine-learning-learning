"""Script intended to get performance
metrics of some selected base-learners,
which will be used to composite the pre-
dictive values of the metadataset.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler 

import pandas as pd
import numpy as np
import os

# Initial setting
DATASET_PATH = "./uci-datasets/"
DATASET_LIST = os.listdir(DATASET_PATH)
OUTPUT_PATH = "./metafeatures/base-learners-perf.metadata"

# Number of folds when cross-validating
CV_FOLDS_NUMBER = 10

# Model evaluation metric used
SCORING_METRIC = "accuracy"

# Which column index is the class of corresponding
# dataset. Default is the last column.
custom_class_index = {
  "segmentation.data" : 0,
  "hepatitis.data" : 0,
  "abalone.data" : 0,
  "horse-colic.data" : 23,
  "echocardiogram.data" : 1,
  "flag.data" : 6,
}

# Prepare models for fitting
model_knn = KNeighborsClassifier(n_neighbors=5)
model_svm_linear = SVC(kernel="linear")
model_svm_rbf = SVC(kernel="rbf", gamma="auto")
model_tree = DecisionTreeClassifier()

model_list = {
  "KNN" : model_knn, 
  "SVMLinear" : model_svm_linear, 
  "SVMGaussian" : model_svm_rbf, 
  "DecisionTree" : model_tree,
}

# Tell which algoriths demands data normalization
# Note: will use Min-Max Normalization
model_data_normalize = ["KNN", "SVMLinear", "SVMGaussian"]

# Output setup
scores = np.zeros((len(DATASET_LIST), 2 * len(model_list)))

# Fit models from datasets
for dataset_index, dataset in enumerate(DATASET_LIST):
  print(dataset_index, "- Processing", dataset, "...")
  data = pd.read_csv(DATASET_PATH + dataset, 
    header=None,
    na_values=["NA", "?"])

  # Fill missing values
  # Median value to numeric values
  median_values = data.median(axis=0, numeric_only=True)
  data.fillna(median_values, inplace=True)

  # Mode value for categorical data
  mode_values = data.mode(axis=0).iloc[0]
  data.fillna(mode_values, inplace=True)

  # Get class label index
  if dataset in custom_class_index:
    label_index = custom_class_index[dataset]
  else:
    label_index = data.shape[1]-1

  ind_attr_indexes = list(range(data.shape[1]))
  ind_attr_indexes.remove(label_index)
  
  # Split data into independent attributes and
  # class label attribute
  attr_independent = data.iloc[:, ind_attr_indexes]
  attr_labels = data.iloc[:, label_index].values

  # One-Hot encoding for categorical data
  attr_independent = pd.get_dummies(attr_independent).astype(np.float64)

  # Normalize dataset
  normalized_attr = MinMaxScaler().fit_transform(attr_independent)

  for classifier_index, classifier_alg_name in enumerate(model_list):
    print("  ", classifier_index, "fitting model from", classifier_alg_name)

    # Get current ML algorithm to fit a model
    classifier_alg = model_list[classifier_alg_name]

    # Check if current dataset demands data normalization
    normalize = classifier_alg_name in model_data_normalize

    score_test = cross_validate(classifier_alg, 
        normalized_attr if normalize else attr_independent,
        attr_labels,
        scoring=SCORING_METRIC,
        cv=CV_FOLDS_NUMBER,
        return_train_score=False)

    # Average metric and time. Here we lose data dispersion
    # information.
    score_test_avg_performance = score_test["test_score"].mean()
    score_test_avg_time = score_test["fit_time"].mean()

    scores[dataset_index, 2*classifier_index] = score_test_avg_performance
    scores[dataset_index, 2*classifier_index + 1] = score_test_avg_time
    

# Final details to produce correct output
output_column_names = []
for model_name in model_list:
  output_column_names += [
    classifier_alg_name + ".avg_performance", 
    classifier_alg_name + ".avg_time"
  ]

scores = pd.DataFrame(scores, 
  columns=output_column_names, 
  index=DATASET_LIST)

# Produce metadata output file
scores.to_csv(OUTPUT_PATH)
