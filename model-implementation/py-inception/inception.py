import typing as t

import numpy
import trax
import trax.fastmath.numpy as np


def inception_module(
    channels_num_out: t.Tuple[int, int, int, int] = (64, 128, 32, 32),
    channels_num_bottleneck: t.Tuple[int, int] = (96, 16),
):
    """Build a Inception module.

    The core idea of the inception module is to perform 1x1, 3x3, 5x5 and
    max pooling at the same input simultaneously, and let the network to
    figure out which is the best option to handle the current input.

    The bottleneck 1x1 conv layers and the 1x1 conv layer after the pooling
    layer are meant just to reduce computational cost, and are not part of
    the core idea of the inception module.
    """
    in_channels = sum(channels_num_out)

    f_num1, f_num2, f_num3, f_num4 = channels_num_out
    b_num2, b_num3 = channels_num_bottleneck

    path1 = trax.layers.Conv(
        filters=f_num1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="VALID",
    )

    path2 = trax.layers.Serial(
        trax.layers.Conv(
            filters=b_num2, kernel_size=(1, 1), strides=(1, 1), padding="VALID"
        ),
        trax.layers.Conv(
            filters=f_num2, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        ),
    )

    path3 = trax.layers.Serial(
        trax.layers.Conv(
            filters=b_num3, kernel_size=(1, 1), strides=(1, 1), padding="VALID"
        ),
        trax.layers.Conv(
            filters=f_num3, kernel_size=(5, 5), strides=(1, 1), padding="SAME"
        ),
    )

    path4 = trax.layers.Serial(
        trax.layers.MaxPool(pool_size=(3, 3), strides=(1, 1), padding="SAME"),
        trax.layers.Conv(
            filters=f_num4, kernel_size=(1, 1), strides=(1, 1), padding="VALID"
        ),
    )

    module = trax.layers.Serial(
        trax.layers.Select(indices=[0, 0, 0, 0], n_in=1),
        trax.layers.Parallel(path1, path2, path3, path4),
        trax.layers.Concatenate(n_items=4),
    )

    return module


def get_data() -> t.Tuple[np.ndarray, ...]:
    X_train = np.load("datasets/train_set_x_orig.npy") / 255.0
    X_test = np.load("datasets/test_set_x_orig.npy") / 255.0

    y_train = np.load("datasets/train_set_y_orig.npy").astype(int).ravel()
    y_test = np.load("datasets/test_set_y_orig.npy").astype(int).ravel()

    classes = np.load("datasets/classes.npy")

    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("Classes:", classes)

    return X_train, X_test, y_train, y_test, classes


def data_gen(X, y, batch_size: int, shuffle: bool = False):
    inds = numpy.arange(X.shape[0])

    if shuffle:
        numpy.random.shuffle(inds)

    cur_ind = 0

    while True:
        X_batch = X[inds[cur_ind : cur_ind + batch_size], :]
        y_batch = y[inds[cur_ind : cur_ind + batch_size]]

        cur_ind += batch_size

        if cur_ind >= inds.size:
            cur_ind %= inds.size

            if shuffle:
                numpy.random.shuffle(inds)

            X_batch_rem = X[inds[:cur_ind], :]
            y_batch_rem = y[inds[:cur_ind]]

            X_batch = np.vstack((X_batch, X_batch_rem))
            y_batch = np.concatenate((y_batch, y_batch_rem))

        yield X_batch, y_batch, np.ones(batch_size)


def _test():
    import sklearn.metrics
    import os

    train = True
    inception_layers = 1
    num_epochs_train = 256
    checkpoint_path = f"inception_model_{inception_layers}"

    X_train, X_eval, y_train, y_eval, classes = get_data()
    num_classes = len(classes)

    full_model = trax.layers.Serial(
        trax.layers.Conv(
            filters=256, kernel_size=(1, 1), strides=(1, 1), padding="SAME"
        ),
        [inception_module() for _ in np.arange(1)],
        trax.layers.Flatten(n_axes_to_keep=1),
        trax.layers.Dense(num_classes),
    )

    if train:
        train_generator = data_gen(X_train, y_train, batch_size=32)
        eval_generator = data_gen(X_eval, y_eval, batch_size=32)

        train_task = trax.supervised.training.TrainTask(
            labeled_data=train_generator,
            loss_layer=trax.layers.CrossEntropyLossWithLogSoftmax(),
            optimizer=trax.optimizers.Adam(0.01),
            n_steps_per_checkpoint=5,
        )

        eval_task = trax.supervised.training.EvalTask(
            labeled_data=eval_generator,
            metrics=[trax.layers.CrossEntropyLossWithLogSoftmax()],
        )

        loop = trax.supervised.training.Loop(
            full_model,
            train_task,
            eval_tasks=eval_task,
            output_dir=checkpoint_path,
        )

        loop.run(n_steps=num_epochs_train)
        loop.save_checkpoint()

    full_model.init_from_file(os.path.join(checkpoint_path, "model.pkl.gz"))

    eval_preds = full_model(X_eval).argmax(axis=1)
    eval_acc = sklearn.metrics.accuracy_score(eval_preds, y_eval)
    print(f"Eval accuracy: {eval_acc:.4f}")


if __name__ == "__main__":
    _test()
