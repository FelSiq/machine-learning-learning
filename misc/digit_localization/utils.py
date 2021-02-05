import typing as t
import glob
import os
import re
import random

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.axes
import matplotlib
import numpy as np
import torch
import torchvision

import config


def plot_instance(
    inst,
    label,
    is_object_threshold: float = 0.6,
    show: bool = True,
    fig_suptitle: t.Optional[str] = None,
):
    fig, ax = plt.subplots(1)
    fig.suptitle(fig_suptitle, fontsize=20)
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


def get_data(
    train_frac: float,
    shuffle_paths: bool = True,
    verbose: bool = True,
    debug: bool = False,
):
    insts_path = glob.glob(os.path.join(config.DATA_DIR, "insts_*.pt"))
    get_target_id_re = re.compile(r"(?<=insts_)([0-9]+)\.pt$")

    if shuffle_paths:
        random.shuffle(insts_path)

    if debug:
        print("Will use only 2 data chunks due to activated debug mode.")
        insts_path = insts_path[:2]

    for i, insts_chunk_path in enumerate(insts_path, 1):
        chunk_id = get_target_id_re.search(insts_chunk_path).group(1)
        target_chunk_path = os.path.join(config.DATA_DIR, f"targets_{chunk_id}.pt")

        X = torch.load(insts_chunk_path)
        y = torch.load(target_chunk_path)

        if debug:
            X = X[:20, ...]
            y = y[:20, ...]

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
            if debug:
                print("Debug mode activated, will use only 20 instances per chunk.")

            print(
                "Chunk filepath:",
                insts_chunk_path,
                target_chunk_path,
                f"(chunk {i} of {len(insts_path)})",
            )
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


def non_max_suppresion(y_preds, iou_threshold: float = 0.1, by_class: bool = True):
    is_object = y_preds[:, 0, ...]
    coords = y_preds[:, [1, 2], ...]
    dims = y_preds[:, [3, 4], ...]
    class_probs = y_preds[:, 5:, ...]

    pos_inds_y = (
        torch.tensor(range(config.NUM_CELLS_VERT))
        .repeat_interleave(config.NUM_CELLS_HORIZ)
        .reshape(config.NUM_CELLS_VERT, config.NUM_CELLS_HORIZ)
    )
    pos_inds_x = torch.tensor(range(config.NUM_CELLS_HORIZ)).repeat(
        config.NUM_CELLS_VERT, 1
    )
    coords = coords.permute(0, 2, 3, 1)
    dims = dims.permute(0, 2, 3, 1)

    coords[..., 0] += pos_inds_y.to(coords.device)
    coords[..., 1] += pos_inds_x.to(coords.device)

    coords = coords.reshape(-1, 2)
    dims = dims.reshape(-1, 2)
    is_object = is_object.reshape(-1)

    half_dims = 0.5 * dims
    boxes = torch.cat((coords - half_dims, coords + half_dims), dim=1)

    if by_class:
        idxs = class_probs.argmax(dim=1).view(-1)
        keep_inds = torchvision.ops.batched_nms(
            boxes=boxes,
            scores=is_object,
            idxs=idxs,
            iou_threshold=iou_threshold,
        )

    else:
        keep_inds = torchvision.ops.nms(
            boxes=boxes,
            scores=is_object,
            iou_threshold=iou_threshold,
        )

    erase_inds = [i for i in range(len(is_object)) if i not in set(keep_inds.tolist())]

    if erase_inds:
        y_preds = y_preds.permute(0, 2, 3, 1)
        y_preds_shape = y_preds.shape
        output_depth = y_preds_shape[-1]
        y_preds = y_preds.reshape(-1, output_depth)
        y_preds[erase_inds, 0] = 0.0
        y_preds = y_preds.reshape(*y_preds_shape)
        y_preds = y_preds.permute(0, 3, 1, 2)

    return y_preds


def _test():
    for X, y in get_data(train_frac=0.9, debug=True):
        pass


if __name__ == "__main__":
    _test()
