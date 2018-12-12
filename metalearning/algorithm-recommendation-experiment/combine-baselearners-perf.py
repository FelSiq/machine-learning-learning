"""Script supposed to combine performance and
execution time metrics into a single metric
for all base learners predictive models. It also
append the final output into the metafeatures
dataset.
"""
from scipy.stats import rankdata
import pandas as pd
import numpy as np
import re

# Script configuration
METADATA_PATH = "./metafeatures/"
PERFORMANCE_METRIC = "accuracy"
METAFEATURES_NAME = "metafeatures-extracted.metadata"
PERFORMANCE_DATA_NAME = "base-learners-perf-" +\
        PERFORMANCE_METRIC + ".metadata"
RANK_MODELS = True
OUTPUT_PATH = METADATA_PATH + "final_combined-" +\
        PERFORMANCE_METRIC + ".metadata"

# Read dataset performance metrics
perf_metadata = pd.read_csv(
    METADATA_PATH +
    PERFORMANCE_DATA_NAME
)

# Combine performance metrics with time spend
# for all base learners using A3R metric
# (See Abdulrahman and Brazdil:
# http://ceur-ws.org/Vol-1201/paper-11.pdf)


def a3r(success_rate_A, success_rate_B,
        time_spend_A, time_spend_B, n=2):
    """Combine Success rate and computational time of
    two base learners algorithms based on ratios between
    the overall performanfe of the two.
    """
    return (success_rate_A / success_rate_B) *\
           (time_spend_B / time_spend_A)**n


out_num_row, out_num_col = perf_metadata.shape
out_num_col //= 2
combined_metrics = np.zeros((out_num_row, out_num_col))

for cur_index_dataset, dataset_perf in \
        enumerate(perf_metadata.iloc[:, 1:].values):

    all_perfs = dataset_perf[:-1:2]
    all_times = dataset_perf[1::2]

    for cur_index_model, cur_metrics in enumerate(zip(all_perfs, all_times)):
        cur_model_perf, cur_model_time = cur_metrics
        cur_model_a3r = a3r(
                cur_model_perf, all_perfs,
                cur_model_time, all_times
        )

        # Remove a3r with itself
        cur_model_a3r_median = np.delete(cur_model_a3r, cur_index_model)

        # Normalize this model a3r for current dataset
        cur_model_a3r_median /= cur_model_a3r_median.sum()

        # Get median a3r of this model for current dataset
        cur_model_a3r_median = np.median(cur_model_a3r_median)

        combined_metrics[cur_index_dataset, cur_index_model] =\
            cur_model_a3r_median


# Rank each dataset row (logically, the lower
# the value the higher the rank, therefore the
# better algorithm performed in the dataset.
if RANK_MODELS:
    ranking_output = np.zeros(combined_metrics.shape)
    for dataset_index, dataset_metric in enumerate(combined_metrics):
        ranking_output[dataset_index, :] =\
            rankdata(-dataset_metric, method="min")
    ranking_output = ranking_output.astype(np.int32)

# Read metafeature dataset
metadata = pd.read_csv(
    METADATA_PATH +
    METAFEATURES_NAME
)

# Prepare output dataframe
regex_get_model_name = re.compile(r"([^\.]+)")
output_column_names = [
    regex_get_model_name.match(model_name).group(1) +
    (".rank" if RANK_MODELS else ".normalized_median_a3r")
    for model_name in perf_metadata.columns[1:-1:2]
]

ranking_output = pd.DataFrame(
    ranking_output if RANK_MODELS else combined_metrics,
    columns=output_column_names,
    index=metadata.index.values
)

# Append combined perf metrics into
# metafeature dataset
metadata[output_column_names] = ranking_output

# Generate output file
metadata.to_csv(OUTPUT_PATH)
