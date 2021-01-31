import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.axes
import matplotlib
import numpy as np

import config


def plot_instance(inst, label):
    fig, ax = plt.subplots(1)

    ax.imshow(inst, cmap="gray")

    for y in range(config.NUM_CELLS_VERT):
        for x in range(config.NUM_CELLS_HORIZ):
            (
                is_object,
                center_y_prop,
                center_x_prop,
                rect_height_prop,
                rect_width_prop,
            ) = label[y, x, :5]

            if np.isclose(is_object, 1.0):
                true_center_y = (y + center_y_prop) * config.CELL_HEIGHT
                true_center_x = (x + center_x_prop) * config.CELL_WIDTH

                rect_anchor_y = (
                    true_center_y - (rect_height_prop * config.CELL_HEIGHT) / 2
                )
                rect_anchor_x = (
                    true_center_x - (rect_width_prop * config.CELL_WIDTH) / 2
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

    plt.show()
