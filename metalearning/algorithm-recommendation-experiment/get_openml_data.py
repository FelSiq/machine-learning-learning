"""Script made for get OpenML 100 datasets."""
import openml
import pandas as pd


def main():
    benchmark = openml.study.get_study("OpenML100", "tasks")

    FULL_PATH = "./openml-data/"
    EXTENSION = ".csv"

    errors = []
    for task_id in benchmark.tasks:
        print("Getting task with id =", task_id, "...", end=" ")

        try:
            task = openml.tasks.get_task(task_id)

        except Exception:
            print("failed.")
            errors.append(task_id)

        else:
            dataset_name = task.get_dataset().name
            print('Got dataset "', dataset_name, '".')

            X, y = task.get_X_and_y()
            X = pd.DataFrame(X)
            X["target"] = y

            output_path = [FULL_PATH, dataset_name, EXTENSION]
            X.to_csv("".join(output_path))

    print("\nFinished. Total of", len(errors),
          "datasets that could not be downloaded:")

    for dataset_id in errors:
        print(dataset_id)


if __name__ == "__main__":
    main()
