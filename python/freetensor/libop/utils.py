def begin_with_0(lst):
    return len(lst) > 0 and lst[0] == 0


def all_minus_one(lst):
    return list(map(lambda x: x - 1, lst))
