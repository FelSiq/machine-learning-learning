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

# Model evaluation metric used
SCORING_METRIC = "f1_micro"
print("Metric used:", SCORING_METRIC)

# Initial datapath setting
DATASET_PATH = "./datasets/openml-datasets/"
DATASET_LIST = os.listdir(DATASET_PATH)
OUTPUT_PATH = "./metafeatures/openml-metafeatures/base-learners-perf-" +\
    SCORING_METRIC + ".metadata"

# Number of folds when cross-validating
CV_FOLDS_NUMBER = 10

# Which column index is the class of corresponding
# dataset. Default is the last column.
CUSTOM_CLASS_INDEX = {
    "segmentation.data": 0,
    "hepatitis.data": 0,
    "abalone.data": 0,
    "horse-colic.data": 23,
    "echocardiogram.data": 1,
    "flag.data": 6,
}

# Prepare models for fitting
model_knn = KNeighborsClassifier(n_neighbors=5)
model_svm_linear = SVC(kernel="linear")
model_svm_rbf = SVC(kernel="rbf", gamma="auto")
model_tree = DecisionTreeClassifier()

MODEL_LIST = {
    "KNN": model_knn,
    "SVMLinear": model_svm_linear,
    "SVMGaussian": model_svm_rbf,
    "DecisionTree": model_tree,
}

# Tell which algoriths demands data normalization
# Note: will use Min-Max Normalization
MODEL_DATA_NORMALIZE_LIST = ["KNN", "SVMLinear", "SVMGaussian"]


def _resolve_output(output_path, dataset_list=None, model_list=None):
    """Load a existing output file or create the model for a new one."""
    try:
        print("Output file found. Loading current one.")
        scores = pd.read_csv(output_path, index_col=0)

    except FileNotFoundError:
        print("No output file found. Creating a empty one...")

        # Output setup
        scores = np.zeros((len(dataset_list), 2 * len(model_list)))

        # Final details to produce correct output
        output_column_names = []
        for classifier_alg_name in MODEL_LIST:
            output_column_names += [
                classifier_alg_name + ".avg_performance",
                classifier_alg_name + ".avg_time",
            ]

        scores = pd.DataFrame(
            scores,
            columns=output_column_names,
            index=DATASET_LIST,
        )

    finally:
        return scores


def get_data(dataset_path):
    """Get metafeatures dataset."""
    data = pd.read_csv(
        dataset_path,
        na_values=["NA", "?"],
    )

    # Fill missing values
    # Median value to numeric values
    median_values = data.median(axis=0, numeric_only=True)
    data.fillna(median_values, inplace=True)

    # Mode value for categorical data
    mode_values = data.mode(axis=0).iloc[0]
    data.fillna(mode_values, inplace=True)

    return data


def get_attr_and_labels(data, dataset_name, custom_class_index):
    """."""
    # Get class label index
    if dataset_name in custom_class_index:
        label_index = custom_class_index[dataset_name]
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

    return attr_independent, attr_labels


def _normalize_attr(attr_independent, dataset_name):
    """Normalize dataset with MinMaxScaler."""
    try:
        normalized_attr = MinMaxScaler().fit_transform(attr_independent)

    except Exception:
        print("Can't normalize dataset", dataset_name,
              ". Will proceed without normalization.")
        normalized_attr = attr_independent

    return normalized_attr


def get_cv_score(
        classifier_alg,
        attr_independent,
        attr_labels,
        scoring="accuracy_score",
        cv=10,
        return_train_score=False):
    """Return cross validation score of given algorithm and dataset."""

    try:
        score_test = cross_validate(
            classifier_alg,
            attr_independent,
            attr_labels,
            scoring=scoring,
            cv=cv,
            return_train_score=return_train_score,
        )

        error = False

    except Exception:

        score_test = {
            "test_score": np.array([-1.0]),
            "fit_time": np.array([-1.0]),
        }

        error = True

    return score_test, error


def _fill_performance(
        scores,
        score_test,
        dataset_index,
        classifier_index):
    """Fill performance in output DataFrame."""
    # Average metric and time. Here we lose data dispersion
    # information.
    score_test_avg_performance = score_test["test_score"]
    score_test_avg_time = score_test["fit_time"]

    score_test_avg_performance = score_test_avg_time[
        np.logical_not(np.isnan(score_test_avg_performance))
    ].mean()
    score_test_avg_time = score_test_avg_time[
        np.logical_not(np.isnan(score_test_avg_time))
    ].mean()

    scores.iloc[
        dataset_index, 2*classifier_index] = score_test_avg_performance
    scores.iloc[
        dataset_index, 2*classifier_index + 1] = score_test_avg_time


def _check_dataset_state(dataset, scores):
    """Check if dataset was proceceed in previous script run."""
    return sum(scores.loc[dataset, :]) != 0.0


def main():
    scores = _resolve_output(OUTPUT_PATH, DATASET_LIST, MODEL_LIST)
    errors = []

    print(scores.head())

    # Fit models from datasets
    for dataset_index, dataset in enumerate(DATASET_LIST):
        if _check_dataset_state(dataset, scores):
            continue

        print(dataset_index, "- Processing", dataset, "...")
        data = get_data(DATASET_PATH + dataset)

        attr_independent, attr_labels = get_attr_and_labels(
                data, dataset, CUSTOM_CLASS_INDEX)

        normalized_attr = _normalize_attr(attr_independent, dataset)

        for classifier_index, classifier_alg_name in enumerate(MODEL_LIST):
            print("    ", classifier_index,
                  "fitting model from", classifier_alg_name)

            # Get current ML algorithm to fit a model
            classifier_alg = MODEL_LIST[classifier_alg_name]

            # Check if current dataset demands data normalization
            normalize = classifier_alg_name in MODEL_DATA_NORMALIZE_LIST

            score_test, error = get_cv_score(
               classifier_alg,
               normalized_attr if normalize else attr_independent,
               attr_labels,
               scoring=SCORING_METRIC,
               cv=CV_FOLDS_NUMBER,
               return_train_score=False,
            )

            if error:
                print("Can't train model", classifier_alg_name,
                      "on dataset", dataset, ".")
                errors.append((dataset, classifier_alg_name))

            _fill_performance(
                scores,
                score_test,
                dataset_index,
                classifier_index,
            )

        # Produce metadata output file
        scores.to_csv(OUTPUT_PATH)

    print("Done. Total of", len(errors), "errors.")
    for err in errors:
        print(err)


if __name__ == "__main__":
    main()
