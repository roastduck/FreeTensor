import ir

# TODO: Currently, a callee function must be in the global scope. Can we support a local scope?


@ir.transform
def g(y):
    ir.declare_var(y, (2,), "float32", "output", "cpu")
    '''nid: S0'''
    y[0] = 2.0
    '''nid: S1'''
    y[1] = 3.0


@ir.transform
def f1(y):
    ir.declare_var(y, (2,), "float32", "output", "cpu")
    g(y)


@ir.transform
def f2(y1, y2):
    ir.declare_var(y1, (2,), "float32", "output", "cpu")
    ir.declare_var(y2, (2,), "float32", "output", "cpu")
    '''nid: C1'''
    g(y1)
    '''nid: C2'''
    g(y2)


def test_basic_call():
    func = ir.lower(f1, ir.CPU())
    print(func)

    with ir.VarDef("y", (2,), "float32", "output", "cpu") as y:
        y[0] = 2.0
        y[1] = 3.0
    std = ir.pop_ast()
    assert std.match(func.body)


def test_called_multiple_times():
    func = ir.lower(f2, ir.CPU())
    print(func)

    with ir.VarDef([("y1", (2,), "float32", "output", "cpu"),
                    ("y2", (2,), "float32", "output", "cpu")]) as (y1, y2):
        y1[0] = 2.0
        y1[1] = 3.0
        y2[0] = 2.0
        y2[1] = 3.0
    std = ir.pop_ast()
    assert std.match(func.body)

    s = ir.Schedule(func)
    assert len(s.find_all(lambda x: x.nid() == "C1.S0")) == 1
    assert len(s.find_all(lambda x: x.nid() == "C1.S1")) == 1
    assert len(s.find_all(lambda x: x.nid() == "C2.S0")) == 1
    assert len(s.find_all(lambda x: x.nid() == "C2.S1")) == 1
