import os
import glob
import re

import torchvision
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
import torch

import config
import utils

OUTPUT_LEN = 256
KEEP_ASPECT_RATIO = True
MIN_INST_DIM = 28
MAX_INST_DIM = 128


def get_file_id() -> int:
    get_cur_file_ind = re.compile(r"[^_]+_(\d+)")

    cur_id = -1

    for path in glob.glob(config.DATA_DIR + "/*"):
        res = get_cur_file_ind.search(path)
        if res is not None:
            cur_id = max(cur_id, int(res.group(1)))

    return cur_id + 1


def _test(plot: int = 0):
    train_data = torchvision.datasets.MNIST(
        root=".",
        train=True,
        download=True,
    )

    X, y = train_data.data, train_data.targets

    def insert_inplace(
        new_inst, top_left_coord_y, top_left_coord_x, inst_height, inst_width, sample
    ):
        new_inst[
            top_left_coord_y : top_left_coord_y + inst_height,
            top_left_coord_x : top_left_coord_x + inst_width,
        ] += sample

        new_inst = np.maximum(new_inst, 255.0)

    gen_insts = np.zeros(
        (OUTPUT_LEN, config.OUTPUT_HEIGHT, config.OUTPUT_WIDTH), dtype=np.float32
    )
    gen_targets = np.zeros(
        (
            OUTPUT_LEN,
            config.TARGET_DEPTH,
            config.NUM_CELLS_VERT,
            config.NUM_CELLS_HORIZ,
        ),
        dtype=np.float32,
    )

    for i in np.arange(OUTPUT_LEN):
        new_inst = np.zeros(
            (config.OUTPUT_HEIGHT, config.OUTPUT_WIDTH), dtype=np.float32
        )
        new_target = np.zeros(
            (
                config.TARGET_DEPTH,
                config.NUM_CELLS_VERT,
                config.NUM_CELLS_HORIZ,
            ),
            dtype=np.float32,
        )
        num_digits = np.random.randint(1, 5)

        for j in np.arange(num_digits):
            sample_ind = np.random.randint(y.size())
            sample = X[sample_ind, ...]
            target = y[sample_ind].item()

            if KEEP_ASPECT_RATIO:
                inst_dim = np.random.randint(MIN_INST_DIM, MAX_INST_DIM + 1)
                inst_height, inst_width = inst_dim, inst_dim

            else:
                inst_height, inst_width = np.random.randint(
                    MIN_INST_DIM, MAX_INST_DIM + 1, size=2
                )

            rotation_angle = np.random.randint(-5, 6)
            sample = torchvision.transforms.functional.rotate(sample, rotation_angle)
            sample = torchvision.transforms.functional.resize(
                sample, (inst_height, inst_width)
            )
            sample = sample.squeeze().numpy()

            top_left_coord_y = np.random.randint(config.OUTPUT_HEIGHT - inst_height + 1)
            top_left_coord_x = np.random.randint(config.OUTPUT_WIDTH - inst_width + 1)

            insert_inplace(
                new_inst,
                top_left_coord_y,
                top_left_coord_x,
                inst_height,
                inst_width,
                sample,
            )

            center_y = top_left_coord_y + inst_height // 2
            center_x = top_left_coord_x + inst_width // 2

            cell_y, in_cell_y = divmod(center_y, config.CELL_HEIGHT)
            cell_x, in_cell_x = divmod(center_x, config.CELL_WIDTH)

            cell_y = int(cell_y)
            cell_x = int(cell_x)

            in_cell_y_prop = in_cell_y / config.CELL_HEIGHT
            in_cell_x_prop = in_cell_x / config.CELL_WIDTH

            assert 0.0 <= in_cell_y_prop <= 1.0, in_cell_y_prop
            assert 0.0 <= in_cell_x_prop <= 1.0, in_cell_x_prop

            in_cell_height_prop = inst_height / config.CELL_HEIGHT
            in_cell_width_prop = inst_width / config.CELL_WIDTH

            assert in_cell_height_prop > 0.0
            assert in_cell_width_prop > 0.0

            # TODO: choose the correct anchor box when NUM_ANCHOR_BOXES > 1
            digit_target = config.NUM_CLASSES * [0.0]
            digit_target[target] = 1.0
            digit_target = [
                1.0,
                in_cell_y_prop,
                in_cell_x_prop,
                in_cell_height_prop,
                in_cell_width_prop,
            ] + digit_target

            new_target[..., cell_y, cell_x] = digit_target

        if plot > 0:
            utils.plot_instance(new_inst, new_target)
            plot -= 1

        gen_insts[i, ...] = new_inst
        gen_targets[i, ...] = new_target

    file_id = get_file_id()

    gen_insts = torch.from_numpy(gen_insts)
    gen_targets = torch.from_numpy(gen_targets)

    torch.save(gen_insts, os.path.join(config.DATA_DIR, f"insts_{file_id}.pt"))
    torch.save(gen_targets, os.path.join(config.DATA_DIR, f"targets_{file_id}.pt"))


if __name__ == "__main__":
    try:
        os.mkdir(config.DATA_DIR)

    except FileExistsError:
        pass

    _test(plot=5)
