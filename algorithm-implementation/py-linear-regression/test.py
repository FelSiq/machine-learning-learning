import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model


def test(model):
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)

    n_splits = 10

    splitter = sklearn.model_selection.KFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    ref = sklearn.linear_model.LinearRegression()

    rmse_train = rmse_eval = 0.0
    rmse_train_ref = rmse_eval_ref = 0.0

    def rmse(a, b):
        return sklearn.metrics.mean_squared_error(a, b, squared=False)

    for inds_train, inds_eval in splitter.split(X, y):
        X_train, X_eval = X[inds_train, :], X[inds_eval, :]
        y_train, y_eval = y[inds_train], y[inds_eval]

        model.fit(X_train, y_train)
        ref.fit(X_train, y_train)

        y_preds = model.predict(X_eval)
        y_preds_ref = ref.predict(X_eval)
        rmse_eval += rmse(y_preds, y_eval)
        rmse_eval_ref += rmse(y_preds_ref, y_eval)

        y_preds = model.predict(X_train)
        y_preds_ref = ref.predict(X_train)
        rmse_train += rmse(y_preds, y_train)
        rmse_train_ref += rmse(y_preds_ref, y_train)

    rmse_eval /= n_splits
    rmse_train /= n_splits
    rmse_eval_ref /= n_splits
    rmse_train_ref /= n_splits

    print(f"(mine) Train RMSE : {rmse_train:.3f}")
    print(f"(mine) Eval RMSE  : {rmse_eval:.3f}")
    print(f"(ref)  Train RMSE : {rmse_train_ref:.3f}")
    print(f"(ref)  Eval RMSE  : {rmse_eval_ref:.3f}")
