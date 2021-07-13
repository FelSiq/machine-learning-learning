import typing as t

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.datasets


class Eigenfaces(sklearn.base.TransformerMixin):
    def __init__(
        self,
        face_shape: t.Tuple[int, int],
        copy: bool = True,
        n_components: t.Optional[t.Union[float, int, str]] = 128,
        iterated_power: t.Union[int, str] = "auto",
        svd_solver: str = "randomized",
        random_state: t.Optional[int] = None,
    ):
        self.pca = sklearn.decomposition.PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=True,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        self.eigenfaces = np.empty(0)
        self.copy = copy
        self.face_shape = face_shape

    def fit(self, X, y=None):
        if self.copy:
            X = np.copy(X).astype(float, copy=False)

        else:
            X = np.asfarray(X)

        self.pca.fit(X)

        self.eigenfaces = self.pca.components_.reshape(
            (self.pca.n_components_, *self.face_shape)
        )

        return self

    def transform(self, X, y=None):
        return self.pca.transform(X)


def plot_samples(X, rows: int = 4, cols: int = 4, random: bool = True):
    fig, axes = plt.subplots(rows, cols)

    if random:
        samples = np.random.randint(X.shape[0], size=rows * cols)

    else:
        samples = np.arange(rows * cols)

    for i in range(rows * cols):
        im = X[samples[i]].reshape((62, 47))
        ax = axes[i // rows][i % cols]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(im, cmap="hot")

    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def load_data(print_shape: bool = True):
    X, y = sklearn.datasets.fetch_lfw_people(min_faces_per_person=25, return_X_y=True)

    if print_shape:
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    return X, y


def _test():
    import sklearn.svm
    import sklearn.model_selection
    import sklearn.metrics
    import scipy.stats
    import pandas as pd

    random_search = False

    X, y = load_data(print_shape=True)
    # plot_samples(X, random=True)

    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=16,
    )

    model = Eigenfaces(
        face_shape=(62, 47), n_components=130, random_state=16, iterated_power=10
    )
    proj_train = model.fit_transform(X_train)
    proj_eval = model.transform(X_eval)

    # plot_samples(model.eigenfaces, random=False)
    classifier = sklearn.svm.SVC(
        C=40000,
        gamma=0.0035,
        kernel="rbf",
        cache_size=2000,
        random_state=16,
    )

    if random_search:
        param_distributions = {
            "C": scipy.stats.uniform(1e3, 1e6),
            "gamma": scipy.stats.loguniform(1e-4, 1e-2),
        }

        classifier = sklearn.model_selection.RandomizedSearchCV(
            classifier,
            param_distributions,
            cv=5,
            n_iter=10,
            n_jobs=-1,
            scoring="f1_weighted",
            random_state=16,
        )

    classifier.fit(proj_train, y_train)

    if random_search:
        print("Best parameters from random search:", classifier.best_params_)

    classifier_preds = classifier.predict(proj_eval)

    baseline = sklearn.svm.SVC(kernel="linear", cache_size=2000)
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_eval)

    classifier_f1 = sklearn.metrics.f1_score(
        classifier_preds, y_eval, average="weighted"
    )
    baseline_f1 = sklearn.metrics.f1_score(baseline_preds, y_eval, average="weighted")

    classifier_accuracy = sklearn.metrics.accuracy_score(classifier_preds, y_eval)
    baseline_accuracy = sklearn.metrics.accuracy_score(baseline_preds, y_eval)

    cls, freqs = np.unique(y_train, return_counts=True)
    print(f"Maj class      : {np.max(freqs / float(np.sum(freqs))):.4f}")
    print(f"Baseline ACC   : {baseline_accuracy:.4f}")
    print(f"Classifier ACC : {classifier_accuracy:.4f}")
    print(f"Baseline F1    : {baseline_f1:.4f}")
    print(f"Classifier F1  : {classifier_f1:.4f}")


if __name__ == "__main__":
    _test()
