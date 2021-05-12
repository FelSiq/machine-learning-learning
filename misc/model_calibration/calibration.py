import typing as t

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import sklearn.linear_model
import sklearn.isotonic
import sklearn.base


class _BaseCalibrator:
    def __init__(
        self,
        base_estimator,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: t.Optional[int] = None,
        return_uncalibrated: bool = False,
        predict_method_name: str = "predict",
        uncalibrated_threshold: float = 0.5,
    ):
        assert int(n_splits) > 0

        self.base_estimator = base_estimator
        self.calibrator_base = None
        self.n_splits = float(n_splits)
        self.return_uncalibrated = return_uncalibrated
        self.uncalibrated_threshold = float(uncalibrated_threshold)

        self._estimator_predict_method_name = predict_method_name
        self._calibrator_predict_method_name = None

        self._splitter = sklearn.model_selection.KFold(
            n_splits=int(n_splits),
            shuffle=shuffle,
            random_state=random_state,
        )
        self.n_splits = int(n_splits)

        self._models = []

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asarray(y, dtype=int)

        assert np.unique(y).size == 2

        self._models = []

        for inds_train, inds_eval in self._splitter.split(X, y):
            X_train, X_eval = X[inds_train, :], X[inds_eval, :]
            y_train, y_eval = y[inds_train], y[inds_eval]

            estimator = sklearn.base.clone(self.base_estimator)
            estimator.fit(X_train, y_train)

            y_preds_uncalibrated = self._predict(
                X_eval, estimator, self._estimator_predict_method_name
            )

            calibrator = sklearn.base.clone(self.calibrator_base)
            calibrator.fit(y_preds_uncalibrated.reshape(-1, 1), y_eval)

            self._models.append((estimator, calibrator))

        return self

    def _predict(self, X, model, method_name):
        pred_method = getattr(model, method_name)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_preds = pred_method(X)

        if y_preds.ndim > 1:
            y_preds = y_preds[:, 1]

        return y_preds

    def predict(self, X):
        X = np.asfarray(X)
        n, _ = X.shape
        y_preds_uncalibrated, y_preds = np.zeros((2, n))

        for estimator, calibrator in self._models:
            cur_y_preds_uncalibrated = self._predict(
                X, estimator, self._estimator_predict_method_name
            )

            cur_y_preds = self._predict(
                cur_y_preds_uncalibrated,
                calibrator,
                self._calibrator_predict_method_name,
            )

            y_preds_uncalibrated += cur_y_preds_uncalibrated
            y_preds += cur_y_preds

        y_preds /= self.n_splits
        y_preds_uncalibrated /= self.n_splits

        if self.return_uncalibrated:
            y_preds_uncalibrated = (
                y_preds_uncalibrated > self.uncalibrated_threshold
            ).astype(int, copy=False)

            return y_preds, y_preds_uncalibrated

        return y_preds


class PlattCalibration(_BaseCalibrator):
    def __init__(
        self,
        base_estimator,
        calibrator_args: t.Dict[str, t.Any] = None,
        *args,
        **kwargs,
    ):
        super(PlattCalibration, self).__init__(base_estimator, *args, **kwargs)
        _calibrator_args = calibrator_args if calibrator_args is not None else {}
        self.calibrator_base = sklearn.linear_model.LogisticRegression(
            **_calibrator_args
        )
        self._calibrator_predict_method_name = "predict_proba"


class IsotonicCalibration(_BaseCalibrator):
    def __init__(
        self,
        base_estimator,
        calibrator_args: t.Dict[str, t.Any] = None,
        *args,
        **kwargs,
    ):
        super(IsotonicCalibration, self).__init__(base_estimator, *args, **kwargs)
        _calibrator_args = calibrator_args if calibrator_args is not None else {}
        self.calibrator_base = sklearn.isotonic.IsotonicRegression(**_calibrator_args)
        self._calibrator_predict_method_name = "predict"


def calibration_plot(
    y_true,
    y_preds_proba,
    y_preds_uncalibrated=None,
    bins: int = 5,
    ax=None,
):
    fig = None

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    thresholds = np.linspace(0, 1, bins + 1)[1:]
    inds = np.digitize(y_preds_proba, thresholds)

    def propagate_vals_to_nan(vals, initial_value=0.0):
        last_valid_val = initial_value
        for i in range(len(vals)):
            if np.isnan(vals[i]):
                vals[i] = last_valid_val

            last_valid_val = vals[i]

    means = np.asfarray([np.mean(y_true[inds == i]) for i in range(bins)])
    means_preds = np.asfarray([np.mean(y_preds_proba[inds == i]) for i in range(bins)])

    propagate_vals_to_nan(means)
    propagate_vals_to_nan(means_preds)

    ax.plot([0, 1], [0, 1], linestyle="--", color="red")
    ax.plot(
        means_preds,
        means,
        color="blue",
        label=None if y_preds_uncalibrated is None else "calibrated",
        lw=2,
    )

    if y_preds_uncalibrated is not None:
        means_uncalibrated = np.asfarray(
            [np.mean(y_preds_uncalibrated[inds == i]) for i in range(bins)]
        )
        ax.plot(
            means_preds,
            means_uncalibrated,
            linestyle="-.",
            color="orange",
            label="uncalibrated",
        )

        ax.legend()

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    return fig, ax


def _test_base(model, ref, name, bins: int = 5):
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.calibration

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=8
    )

    model.fit(X_train, y_train)
    y_preds, y_preds_uncalibrated = model.predict(X_test)
    sk_true, sk_preds = sklearn.calibration.calibration_curve(
        y_test, y_preds, n_bins=bins
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    calibration_plot(y_test, y_preds, y_preds_uncalibrated, ax=ax1, bins=bins)
    ax1.set_title(f"{name} (mine)")
    ax1.scatter(sk_preds, sk_true, label="(ref) sklearn", color="purple", s=64)
    ax1.legend()

    ref.fit(X_train, y_train)
    y_preds_ref = ref.predict_proba(X_test)[:, 1]
    sk_true_ref, sk_preds_ref = sklearn.calibration.calibration_curve(
        y_test, y_preds_ref, n_bins=bins
    )

    calibration_plot(y_test, y_preds_ref, ax=ax2, bins=bins)
    ax2.set_title(f"{name} (sklearn)")
    ax2.scatter(sk_preds_ref, sk_true_ref, label="(ref) sklearn", color="purple", s=64)
    ax2.legend()

    plt.show()


def _test_01():
    import sklearn.svm
    import sklearn.calibration

    svc = sklearn.svm.SVC()
    model_platt = PlattCalibration(
        svc, return_uncalibrated=True, predict_method_name="decision_function"
    )
    ref_platt = sklearn.calibration.CalibratedClassifierCV(svc, method="sigmoid", cv=5)
    name = "Platt Calibration"
    _test_base(model_platt, ref_platt, name)


def _test_02():
    import sklearn.svm
    import sklearn.calibration

    svc = sklearn.svm.SVC()
    model_isotonic = IsotonicCalibration(
        svc, return_uncalibrated=True, predict_method_name="decision_function"
    )
    ref_isotonic = sklearn.calibration.CalibratedClassifierCV(
        svc, method="isotonic", cv=5
    )
    name = "Isotonic Calibration"
    _test_base(model_isotonic, ref_isotonic, name)


if __name__ == "__main__":
    _test_01()
    _test_02()
