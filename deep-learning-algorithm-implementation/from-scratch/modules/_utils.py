def all_positive(vals):
    a = not isinstance(vals, int) or vals > 0

    b = not hasattr(vals, "__len__") or all(map(lambda x: x > 0, vals))

    return a and b
