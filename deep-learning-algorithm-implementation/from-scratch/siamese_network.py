import numpy as np

import losses
import modules
import optimizers


class SiameseNetwork(modules.BaseModel):
    def __init__(self, dim_in: int, dim_hidden: int, dim_embed: int):
        super(SiameseNetwork, self).__init__()

        self.weights = modules.Sequential(
            [
                modules.Flatten(),
                modules.Linear(dim_in, dim_hidden, include_bias=False),
                modules.BatchNorm1d(dim_hidden),
                modules.ReLU(inplace=True),
                modules.Linear(dim_hidden, dim_embed, include_bias=False),
                modules.BatchNorm1d(dim_embed),
                modules.ReLU(inplace=True),
            ]
        )

        self.sim = modules.CosineSimilarity()

        self.register_layers(self.weights, self.sim)

    def __call__(self, X, Y):
        return self.forward(X, Y)

    def forward(self, X, Y):
        X_emb = self.weights(X)
        Y_emb = self.weights(Y)
        out = self.sim(X_emb, Y_emb)
        return out

    def backward(self, dout):
        dX_emb, dY_emb = self.sim.backward(dout)
        dY = self.weights.backward(dY_emb)
        dX = self.weights.backward(dX_emb)
        return dX, dY


def _test():
    import sklearn.datasets
    import tqdm.auto

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

    partitions, min_freq, batch_shape = partition_by_class(X, y)
    num_batches = min_freq // 2

    train_epochs = 39
    lr = 1e-3
    batch_size = 32

    model = SiameseNetwork(8 * 8, 32, 16)
    optim = optimizers.Nadam(model.parameters, learning_rate=lr)
    criterion = losses.TripletLoss(margin=0.4)

    for epoch in np.arange(1, 1 + train_epochs):
        loss_train = loss_eval = 0.0
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
            loss_train += loss_train

            model.eval()
            loss_eval += loss_train

            it += 1

        loss_train /= it
        loss_eval /= it

        print(f"{epoch:<4} - loss train: {loss_train:.3f}, loss eval: {loss_eval:.3f}")


if __name__ == "__main__":
    _test()
