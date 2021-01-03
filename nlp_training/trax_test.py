import numpy
import trax
import trax.fastmath.numpy as np
import trax.supervised
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing


X, y = sklearn.datasets.load_iris(return_X_y=True)

X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, stratify=y
)


scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)


def data_generator(X, y, n_splits: int = 10):
    while True:
        splitter = sklearn.model_selection.ShuffleSplit(n_splits=n_splits)

        for _, inds in splitter.split(X, y):
            yield X[inds, :], y[inds], np.ones_like(inds)


model = trax.layers.Serial(
    trax.layers.Dense(4),
    trax.layers.Relu(),
    trax.layers.Dense(5),
    trax.layers.Relu(),
    trax.layers.LogSoftmax(),
)


train_task = trax.supervised.training.TrainTask(
    labeled_data=data_generator(X_train, y_train),
    loss_layer=trax.layers.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=16,
)

eval_task = trax.supervised.training.EvalTask(
    labeled_data=data_generator(X_eval, y_eval),
    metrics=(trax.layers.CrossEntropyLoss(), trax.layers.Accuracy()),
)


train_loop = trax.supervised.training.Loop(
    model, train_task, eval_tasks=eval_task, output_dir="trax_eval_dir"
)
train_loop.run(n_steps=128)
