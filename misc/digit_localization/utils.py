import typing as t
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.axes
import matplotlib
import numpy as np
import torch

import config


def plot_instance(
    inst,
    label,
    is_object_threshold: float = 0.6,
    show: bool = True,
    fig_suptitle: t.Optional[str] = None,
):
    fig, ax = plt.subplots(1)
    fig.suptitle(fig_suptitle)
    ax.imshow(inst, cmap="gray")
    ax.grid(color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(0, config.OUTPUT_WIDTH, config.CELL_WIDTH))
    ax.set_yticks(np.arange(0, config.OUTPUT_HEIGHT, config.CELL_HEIGHT))

    for y in range(config.NUM_CELLS_VERT):
        for x in range(config.NUM_CELLS_HORIZ):
            (
                is_object,
                center_y_prop,
                center_x_prop,
                rect_height_prop,
                rect_width_prop,
            ) = label[:5, y, x]

            if is_object >= is_object_threshold:
                true_center_y = (y + center_y_prop) * config.CELL_HEIGHT
                true_center_x = (x + center_x_prop) * config.CELL_WIDTH

                rect_anchor_y = true_center_y - 0.5 * (
                    rect_height_prop * config.CELL_HEIGHT
                )
                rect_anchor_x = true_center_x - 0.5 * (
                    rect_width_prop * config.CELL_WIDTH
                )

                rect_width = rect_height_prop * config.CELL_HEIGHT
                rect_height = rect_width_prop * config.CELL_WIDTH

                rect = matplotlib.patches.Rectangle(
                    (rect_anchor_x, rect_anchor_y),
                    rect_width,
                    rect_height,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2,
                )
                ax.add_patch(rect)

                ax.scatter(true_center_x, true_center_y, color="red", lw=0.5)

    if show:
        plt.show()

    return fig, ax


def get_data(train_frac: float, verbose: bool = True):
    insts_path = sorted(glob.glob(os.path.join(config.DATA_DIR, "insts_*.pt")))
    target_path = sorted(glob.glob(os.path.join(config.DATA_DIR, "targets_*.pt")))

    for insts_chunk_path, target_chunk_path in zip(insts_path, target_path):
        X = torch.load(insts_chunk_path)
        y = torch.load(target_chunk_path)

        if X.ndim == 3:
            # Note: add channel dimension (following channel-first convention)
            X = X.unsqueeze(1)

        train_size = int(train_frac * len(y))

        X_train = X[:train_size, ...]
        y_train = y[:train_size, ...]
        X_eval = X[train_size:, ...]
        y_eval = y[train_size:, ...]

        assert len(X_train) == len(y_train)
        assert len(X_eval) == len(y_eval)

        if verbose:
            print("Chunk filepath:", insts_chunk_path, target_chunk_path)
            print("Number of train instances :", len(y_train))
            print("Number of eval instances  :", len(y_eval))
            print("Shape of train instances  :", X_train.shape)
            print("Shape of eval instances   :", X_eval.shape)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        eval_dataset = torch.utils.data.TensorDataset(X_eval, y_eval)

        yield train_dataset, eval_dataset

        del train_dataset, eval_dataset
        del X_train, y_train, X_eval, y_eval
        del X, y
