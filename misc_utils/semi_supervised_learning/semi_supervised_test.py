import numpy as np
import sklearn.datasets
import sklearn.ensemble
import sklearn.calibration
import sklearn.semi_supervised
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics


def _test():
    np.random.seed(16)

    semisupervised = True

    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        shuffle=True,
        random_state=16,
        test_size=0.3,
    )

    unsupervised_inds = np.random.random(y_train.size) <= 0.7
    y_train[unsupervised_inds] = -1

    model = sklearn.ensemble.RandomForestClassifier(10)
    model = sklearn.calibration.CalibratedClassifierCV(model)

    if semisupervised:
        model = sklearn.semi_supervised.SelfTrainingClassifier(
            model, max_iter=None, criterion="threshold"
        )

    else:
        X_train = X_train[~unsupervised_inds]
        y_train = y_train[~unsupervised_inds]

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    acc = sklearn.metrics.accuracy_score(y_preds, y_test)
    print(f"Test acc: {acc:.3f}")


if __name__ == "__main__":
    _test()
