import typing as t

import numpy as np


ShapeType = t.Union[int, t.Tuple[int, int]]


def _solve_shape(shape: ShapeType) -> t.Tuple[int, int]:
    if np.isscalar(shape):
        return int(shape), int(shape)

    return tuple(shape)


def conv_out_size(
    shape_input: ShapeType,
    shape_filter: ShapeType,
    shape_stride: ShapeType,
    shape_padding: ShapeType,
) -> ShapeType:
    inp_h, inp_w = _solve_shape(shape_input)
    fil_h, fil_w = _solve_shape(shape_filter)
    str_h, str_w = _solve_shape(shape_stride)
    pad_h, pad_w = _solve_shape(shape_padding)

    out_h = 1 + (inp_h + 2 * pad_h - fil_h) // str_h
    out_w = 1 + (inp_w + 2 * pad_w - fil_w) // str_w

    return (out_h, out_w)


def _test():
    assert conv_out_size(5, 3, 1, 1) == (5, 5)
    assert conv_out_size(5, 3, 2, 0) == (2, 2)
    assert conv_out_size(5, 3, 2, 1) == (3, 3)


if __name__ == "__main__":
    _test()
