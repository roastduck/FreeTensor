def begin_with_0(lst):
    return len(lst) > 0 and lst[0] == 0


def all_minus_one(lst):
    return list(map(lambda x: x - 1, lst))


def circular_axis(axis, ndim):
    return axis if axis >= 0 else ndim + axis
