import typing as t

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


class ROCAUC:
    def __init__(self):
        self.auc = 0.0

    def fit(self, y_true: np.ndarray, probs_pred: np.ndarray):
        probs_pred = np.asfarray(probs_pred).ravel()
        y_true = np.asarray(y_true, dtype=int).ravel()

        assert probs_pred.shape == y_true.shape

        tp, fp, fn, tn = np.zeros((4, 1 + probs_pred.size), dtype=int)
        probs_pred = np.hstack((-1.0, probs_pred, 2.0))
        probs_sorted_ind = np.argsort(probs_pred)
        thresholds = probs_pred[probs_sorted_ind]

        # At the start, threshold = 0: everyone is positive
        fp[0] = int(np.sum(1 - y_true))
        tp[0] = int(np.sum(y_true))
        fn[0] = 0
        tn[0] = 0

        for i in np.arange(y_true.size):
            ind_cur = probs_sorted_ind[i + 1] - 1

            tp[i + 1] = tp[i] - y_true[ind_cur]
            fp[i + 1] = fp[i] - (1 - y_true[ind_cur])

            fn[i + 1] = fn[i] + y_true[ind_cur]
            tn[i + 1] = tn[i] + (1 - y_true[ind_cur])

        # recall    = correctly_true_positives / total_true_positives
        #           = tp / (tp + fn)

        # precision =  correctly_true_positives / total_predicted_positives
        #           = tp / (tp + fp)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tpr = tpr[::-1]
        fpr = fpr[::-1]

        self.auc = scipy.integrate.trapezoid(y=tpr, x=fpr)

        return fpr, tpr


def plot_roc(
    fpr: t.Sequence[float],
    tpr: t.Sequence[float],
    auc: t.Optional[float] = None,
    ax=None,
):
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    if auc is not None:
        plt.title(f"AUC: {auc:.4f}")


def _test():
    import sklearn.linear_model
    import sklearn.datasets
    import sklearn.preprocessing
    import sklearn.metrics

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X) + 16 * np.random.randn(*X.shape)

    model = sklearn.linear_model.LogisticRegression()
    model.fit(X, y)
    probs_pred = model.predict_proba(X)[:, 1]

    roc_auc = ROCAUC()
    fpr, tpr = roc_auc.fit(y, probs_pred)
    auc = roc_auc.auc
    plot_roc(fpr, tpr, auc)

    sk_fpr, sk_tpr, thres = sklearn.metrics.roc_curve(y, probs_pred)
    sk_auc = sklearn.metrics.roc_auc_score(y, probs_pred)
    plot_roc(sk_fpr, sk_tpr, sk_auc)

    plt.show()


if __name__ == "__main__":
    _test()
