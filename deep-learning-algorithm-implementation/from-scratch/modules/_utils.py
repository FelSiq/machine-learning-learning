def all_positive(vals):
    a = not isinstance(vals, int) or vals > 0
    b = not hasattr(vals, "__len__") or all(map(lambda x: x > 0, vals))
    return a and b


def all_nonzero(vals):
    a = not isinstance(vals, int) or vals != 0
    b = not hasattr(vals, "__len__") or all(map(lambda x: x != 0, vals))
    return a and b


def replicate(vals, n):
    if hasattr(vals, "__len__"):
        assert len(vals) == n
        return tuple(vals)

    return tuple([vals] * n)


def collapse(items, cls):
    if isinstance(items, cls):
        return [items]

    cur_items = []

    for item in items:
        cur_items.extend(collapse(item, cls))

    return cur_items
