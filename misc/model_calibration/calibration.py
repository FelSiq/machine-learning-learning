import typing as t

import numpy as np
import sklearn.model_selection
import sklearn.linear_model
import sklearn.isotonic
import matplotlib.pyplot as plt


class _BaseCalibrator:
    def __init__(
        self,
        base_estimator,
        eval_frac: float = 0.2,
        shuffle: bool = True,
        random_state: t.Optional[int] = None,
        return_uncalibrated: bool = False,
        predict_method_name: str = "predict",
        uncalibrated_threshold: float = 0.5,
    ):
        assert 1.0 > float(eval_frac) > 0.0

        self.base_estimator = base_estimator
        self.calibrator = None
        self.eval_frac = float(eval_frac)
        self.shuffle = shuffle
        self.random_state = random_state
        self.return_uncalibrated = return_uncalibrated
        self.uncalibrated_threshold = float(uncalibrated_threshold)

        self._predict_method_name = predict_method_name
        self._pred_method = None

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asarray(y, dtype=int)

        assert np.unique(y).size == 2

        X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
            X,
            y,
            test_size=self.eval_frac,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        self.base_estimator.fit(X_train, y_train)

        self._pred_method = getattr(self.base_estimator, self._predict_method_name)
        y_preds_uncalibrated = self._predict_uncalibrated(X_eval)
        self.calibrator.fit(y_preds_uncalibrated.reshape(-1, 1), y_eval)

        return self

    def _predict_uncalibrated(self, X):
        y_preds_uncalibrated = self._pred_method(X)

        if y_preds_uncalibrated.ndim > 1:
            y_preds_uncalibrated = y_preds_uncalibrated[:, 1]

        return y_preds_uncalibrated

    def predict(self, X):
        y_preds_uncalibrated = self._predict_uncalibrated(X)
        y_preds = self.calibrator.predict_proba(y_preds_uncalibrated.reshape(-1, 1))

        if y_preds.ndim > 1:
            y_preds = y_preds[:, 1]

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
        **kwargs
    ):
        super(PlattCalibration, self).__init__(base_estimator, *args, **kwargs)
        _calibrator_args = calibrator_args if calibrator_args is not None else {}
        self.calibrator = sklearn.linear_model.LogisticRegression(**_calibrator_args)


class IsotonicCalibration(_BaseCalibrator):
    def __init__(
        self,
        base_estimator,
        calibrator_args: t.Dict[str, t.Any] = None,
        *args,
        **kwargs
    ):
        super(PlattCalibration, self).__init__(base_estimator, *args, **kwargs)
        _calibrator_args = calibrator_args if calibrator_args is not None else {}
        self.calibrator = sklearn.isotonic.IsotonicRegression(**_calibrator_args)


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


def _test():
    import sklearn.calibration
    import sklearn.datasets
    import sklearn.svm
    import sklearn.model_selection
    import sklearn.calibration

    bins = 10

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=8
    )
    svc = sklearn.svm.SVC()

    model_platt = PlattCalibration(
        svc, return_uncalibrated=True, predict_method_name="decision_function"
    )
    model_platt.fit(X_train, y_train)
    y_preds, y_preds_uncalibrated = model_platt.predict(X_test)
    sk_true, sk_preds = sklearn.calibration.calibration_curve(
        y_test, y_preds, n_bins=bins
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    calibration_plot(y_test, y_preds, y_preds_uncalibrated, ax=ax1, bins=bins)
    ax1.set_title("Platt Scaling (mine)")
    ax1.scatter(sk_preds, sk_true, label="(ref) sklearn", color="purple", s=64)
    ax1.legend()

    ref_platt = sklearn.calibration.CalibratedClassifierCV(svc, method="sigmoid", cv=2)
    ref_platt.fit(X_train, y_train)
    y_preds_ref = ref_platt.predict_proba(X_test)[:, 1]
    sk_true_ref, sk_preds_ref = sklearn.calibration.calibration_curve(
        y_test, y_preds_ref, n_bins=bins
    )

    calibration_plot(y_test, y_preds_ref, ax=ax2, bins=bins)
    ax2.set_title("Platt Scaling (sklearn)")
    ax2.scatter(sk_preds_ref, sk_true_ref, label="(ref) sklearn", color="purple", s=64)
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    _test()
