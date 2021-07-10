import numpy as np

import losses
import modules
import optimizers


class SiameseNetwork(modules.BaseModel):
    def __init__(
        self, dim_in: int, dim_hidden: int, dim_embed: int, dropout: float = 0.2
    ):
        super(SiameseNetwork, self).__init__()

        self.weights = modules.Sequential(
            [
                modules.Conv2d(1, 16, 3, include_bias=False),
                modules.BatchNorm2d(16),
                modules.ReLU(inplace=True),
                modules.SpatialDropout(dropout),
                modules.Conv2d(16, 16, 3, include_bias=False),
                modules.BatchNorm2d(16),
                modules.ReLU(inplace=True),
                modules.SpatialDropout(dropout),
                modules.Flatten(),
                modules.Linear(4 * 4 * 16, dim_hidden, include_bias=False),
                modules.BatchNorm1d(dim_hidden),
                modules.ReLU(inplace=True),
                modules.Dropout(dropout),
                modules.Linear(dim_hidden, dim_embed, include_bias=False),
                modules.BatchNorm1d(dim_embed),
                modules.ReLU(inplace=True),
            ]
        )

        self.sim_pairwise = modules.CosineSimilarity()
        self.sim_rowwise = modules.CosineSimilarity(pairwise=False)

        self.register_layers(self.weights, self.sim_pairwise, self.sim_rowwise)

    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        X_emb = self.weights(X)
        Y_emb = self.weights(Y)

        if not self.frozen:
            out = self.sim_pairwise(X_emb, Y_emb)
        else:

            out = self.sim_rowwise(X_emb, Y_emb)

        return out

    def backward(self, dout):
        dX_emb, dY_emb = self.sim_pairwise.backward(dout)
        dY = self.weights.backward(dY_emb)
        dX = self.weights.backward(dX_emb)
        return dX, dY


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import tqdm.auto

    np.random.seed(16)

    def partition_by_class(X, y):
        partitions = {}
        classes, freqs = np.unique(y, return_counts=True)

        min_freq = min(freqs)
        min_freq = 2 * (min_freq // 2)

        for cls in classes:
            cls_inds = y == cls
            partitions[cls] = X[cls_inds, :]

        batch_shape = (min_freq * classes.size // 2, *X.shape[1:])

        return partitions, min_freq, batch_shape

    def prepare_batches(partitions, min_freq, batch_shape):
        train_A = np.empty(batch_shape, dtype=float)
        train_B = np.empty(batch_shape, dtype=float)

        n_cls = len(partitions)

        for i, (cls, X_cls) in enumerate(partitions.items()):
            inds = np.arange(len(X_cls))
            np.random.shuffle(inds)
            inds = inds[:min_freq]
            X_cls = X_cls[inds, :]
            X_cls_A, X_cls_B = np.split(X_cls, 2, axis=0)

            train_A[i::n_cls, ...] = X_cls_A
            train_B[i::n_cls, ...] = X_cls_B

        train_A = train_A.reshape(-1, n_cls, *train_A.shape[1:])
        train_B = train_B.reshape(-1, n_cls, *train_B.shape[1:])

        return zip(train_A, train_B)

    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8, 1)
    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    if len(X_eval) % 2 == 1:
        X_eval = X_eval[:-1, ...]

    batch_A_eval, batch_B_eval = np.split(X_eval, 2, axis=0)
    y_eval = np.equal(*np.split(y_eval, 2, axis=0)).astype(int, copy=False)

    partitions, min_freq, batch_shape = partition_by_class(X_train, y_train)
    num_batches = min_freq // 2

    train_epochs = 39
    lr = 1e-3

    model = SiameseNetwork(8 * 8, 32, 16)
    optim = optimizers.Nadam(model.parameters, learning_rate=lr)
    criterion = losses.TripletLoss(margin=0.4)

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = total_eval_recall = total_eval_precision = 0.0
        it = 0

        batches = prepare_batches(partitions, min_freq, batch_shape)

        for batch_A, batch_B in tqdm.auto.tqdm(batches, total=num_batches):
            model.train()
            optim.zero_grad()
            sim_mat = model(batch_A, batch_B)
            loss_train, loss_grad = criterion(sim_mat)
            model.backward(loss_grad)
            optim.clip_grads_norm()
            optim.step()
            total_loss_train += loss_train

            model.eval()
            y_preds = model(batch_A_eval, batch_B_eval).reshape(y_eval.shape)
            y_preds = (y_preds >= 0.6).astype(int, copy=False)

            recall_eval = sklearn.metrics.recall_score(y_preds, y_eval)
            total_eval_recall += recall_eval

            precision_eval = sklearn.metrics.precision_score(y_preds, y_eval)
            total_eval_precision += precision_eval

            it += 1

        total_loss_train /= it
        total_eval_recall /= it
        total_eval_precision /= it

        print(f"loss train     : {total_loss_train:.3f}")
        print(f"recall eval    : {total_eval_recall:.3f}")
        print(f"precision eval : {total_eval_precision:.3f}")


if __name__ == "__main__":
    _test()
