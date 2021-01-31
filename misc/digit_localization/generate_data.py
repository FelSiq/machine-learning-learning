import torchvision
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt

import config
import utils

OUTPUT_DIR = "./data"
OUTPUT_LEN = 3

KEEP_ASPECT_RATIO = True
MIN_INST_DIM = 28
MAX_INST_DIM = 128


def _test():
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

    for i in np.arange(OUTPUT_LEN):
        new_inst = np.zeros((config.OUTPUT_HEIGHT, config.OUTPUT_WIDTH))
        new_label = np.zeros(
            (
                config.NUM_CELLS_VERT,
                config.NUM_CELLS_HORIZ,
                config.NUM_ANCHOR_BOXES * (5 + config.NUM_CLASSES),
            )
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

            digit_label = config.NUM_CLASSES * [0.0]
            digit_label[target] = 1.0
            digit_label = [
                1.0,
                in_cell_y_prop,
                in_cell_x_prop,
                in_cell_height_prop,
                in_cell_width_prop,
            ] + digit_label

            new_label[cell_y, cell_x] = digit_label

        utils.plot_instance(new_inst, new_label)


if __name__ == "__main__":
    _test()
