import ir


def test_type1_basic():
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 1
        y[()] = 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_type1_before_read():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = 1
            b[()] = x[()]
            y1[()] = b[()] * 2
            y2[()] = b[()] * 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = x[()]
            y1[()] = b[()] * 2
            y2[()] = b[()] * 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_type1_one_then_many():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        y[0] = 1
        with ir.For("i", 0, 4) as i:
            y[i] = i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i
    std = ir.pop_ast()

    assert std.match(ast)


def test_type1_many_then_one_no_remove():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i
        y[0] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i
        y[0] = 1
    std = ir.pop_ast()

    assert std.match(ast)


def test_type1_write_then_reduce():
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 1
        y[()] = y[()] + 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_type1_write_then_reduce_expr_modified_no_remove():
    with ir.VarDef([("y", (), "int32", "output", "cpu"),
                    ("z", (), "int32", "inout", "cpu")]) as (y, z):
        y[()] = z[()]
        z[()] = z[()] + 1
        y[()] = y[()] + 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("y", (), "int32", "output", "cpu"),
                    ("z", (), "int32", "inout", "cpu")]) as (y, z):
        y[()] = z[()]
        z[()] = z[()] + 1
        y[()] = y[()] + 2
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_reduce_then_reduce():
    with ir.VarDef("y", (), "int32", "inout", "cpu") as y:
        y[()] = y[()] + 1
        y[()] = y[()] + 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (), "int32", "inout", "cpu") as y:
        y[()] = y[()] + 3
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_write_then_multiple_reduces():
    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 1
        y[()] = y[()] + 2
        y[()] = y[()] + 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 6
    std = ir.pop_ast()

    assert std.match(ast)


def test_type1_write_then_loop_then_reduce_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ir.For("i", 0, 5) as i:
            y[()] = y[()] + i
        y[()] = y[()] + x[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ir.For("i", 0, 5) as i:
            y[()] = y[()] + i
        y[()] = y[()] + x[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_read_by_following_write_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
        y[()] = y[()] * y[()]
        y[()] = y[()] * y[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
        y[()] = y[()] * y[()]
        y[()] = y[()] * y[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_later_store():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[()] > 0):
            y[()] = x[()]
        y[()] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_later_reduce_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[()] > 0):
            y[()] = x[()]
        y[()] += 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[()] > 0):
            y[()] = x[()]
        y[()] += 1
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_earlier_store_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
        with ir.If(x[()] > 0):
            y[()] = x[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
        with ir.If(x[()] > 0):
            y[()] = x[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_earlier_reduce_no_remove():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
        with ir.If(x[()] > 0):
            y[()] += x[()]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
        with ir.If(x[()] > 0):
            y[()] += x[()]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)


def test_type2_inner_loop():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[i] = x[i] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] * 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_type2_outer_loop():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 8) as j:
                y[j] = x[j] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("j", 0, 8) as j:
            y[j] = x[j] * 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_type2_used_no_remove():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("z", (4,), "int32", "output", "cpu"),
                    ("w", (4,), "int32", "output", "cpu")]) as (x, z, w):
        with ir.VarDef("y", (1,), "int32", "cache", "cpu") as y:
            with ir.For("i", 0, 4) as i:
                y[0] = x[i] * 2
                z[i] = y[0] + 1
                w[i] = y[0] + 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("z", (4,), "int32", "output", "cpu"),
                    ("w", (4,), "int32", "output", "cpu")]) as (x, z, w):
        with ir.VarDef("y", (1,), "int32", "cache", "cpu") as y:
            with ir.For("i", 0, 4) as i:
                y[0] = x[i] * 2
                z[i] = y[0] + 1
                w[i] = y[0] + 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_type2_dynamic():
    with ir.VarDef([("n", (), "int32", "input", "byvalue"),
                    ("m", (), "int32", "input", "byvalue")]) as (n, m):
        with ir.Assert(ir.l_and(n[()] > 0, m[()] > 0)):
            with ir.VarDef([
                ("x", (n[()],), "int32", "input", "cpu"),
                ("y", (n[()],), "int32", "output", "cpu"),
            ]) as (x, y):
                with ir.For("i", 0, n[()]) as i:
                    with ir.For("j", 0, m[()]) as j:
                        y[i] = x[i] * 2
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("n", (), "int32", "input", "byvalue"),
                    ("m", (), "int32", "input", "byvalue")]) as (n, m):
        with ir.Assert(ir.l_and(n[()] > 0, m[()] > 0)):
            with ir.VarDef([
                ("x", (n[()],), "int32", "input", "cpu"),
                ("y", (n[()],), "int32", "output", "cpu"),
            ]) as (x, y):
                with ir.For("i", 0, n[()]) as i:
                    y[i] = x[i] * 2
    std = ir.pop_ast()

    assert std.match(ast)


def test_cross_var_def():
    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, y1, y2):
        with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = x1[()] + 1
            y1[()] = b[()] * 2
            y2[()] = b[()] * 3
        with ir.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = x2[()] + 2
            y1[()] += b[()] * 2
            y2[()] += b[()] * 3
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, y1, y2):
        with ir.VarDef("b1", (), "int32", "cache", "cpu") as b1:
            b1[()] = x1[()] + 1
            with ir.VarDef("b2", (), "int32", "cache", "cpu") as b2:
                b2[()] = x2[()] + 2
                y1[()] = b1[()] * 2 + b2[()] * 2
                y2[()] = b1[()] * 3 + b2[()] * 3
    std = ir.pop_ast()

    assert std.match(ast)


def test_same_parent_but_dep_and_circular_dependency_on_init():
    with ir.VarDef([("f", (10,), "float32", "output", "cpu"),
                    ("u", (10,), "float32", "cache", "cpu")]) as (f, u):
        with ir.For("l", 0, 10) as l:
            with ir.For("j", 0, 10) as j:
                f[j] = 0
                with ir.For("k", 0, 10) as k:
                    f[j] += u[j]
                    f[j] += 1
                u[j] = f[j]
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([("f", (10,), "float32", "output", "cpu"),
                    ("u", (10,), "float32", "cache", "cpu")]) as (f, u):
        with ir.For("l", 0, 10) as l:
            with ir.For("j", 0, 10) as j:
                f[j] = 0
                with ir.For("k", 0, 10) as k:
                    f[j] += u[j] + 1
                u[j] = f[j]
    std = ir.make_reduction(ir.pop_ast())

    assert std.match(ast)
