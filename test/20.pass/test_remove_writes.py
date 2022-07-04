import freetensor as ft


def test_type1_write_then_write():
    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 1
        y[()] = 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_write_then_write_across_loops():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i + 1
        with ft.For("i", 0, 4) as i:
            y[i] = i + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_before_read():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = 1
            b[()] = x[()]
            y1[()] = b[()] * 2
            y2[()] = b[()] * 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x, y1, y2):
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = x[()]
            y1[()] = b[()] * 2
            y2[()] = b[()] * 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_one_then_many():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        y[0] = 1
        with ft.For("i", 0, 4) as i:
            y[i] = i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_one_then_many_reduce_no_remove():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        y[0] = 1
        with ft.For("i", 0, 4) as i:
            y[i] += i
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        y[0] = 1
        with ft.For("i", 0, 4) as i:
            y[i] += i
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_many_then_ones():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
        y[0] = 0
        y[1] = 1
        y[2] = 2
        y[3] = 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        y[0] = 0
        y[1] = 1
        y[2] = 2
        y[3] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_many_then_ones_reduce():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
        y[0] += 1
        y[1] += 2
        y[2] += 3
        y[3] += 4
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        y[0] = 2
        y[1] = 3
        y[2] = 4
        y[3] = 5
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_many_then_one_no_remove():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
        y[0] = 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
        y[0] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_repeated_then_one():
    with ft.VarDef("y", (1,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[0] = i
        y[0] = 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (1,), "int32", "output", "cpu") as y:
        y[0] = 1
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_write_then_reduce():
    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 1
        y[()] = y[()] + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_write_then_reduce_across_loops():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i + 1
        with ft.For("i", 0, 4) as i:
            y[i] += 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i + 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_write_then_reduce_across_loops_different_indices():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 2, 6) as i:
            y[i - 2] = i + 1
        with ft.For("i", 0, 4) as i:
            y[i] += 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i + 5
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_write_then_reduce_expr_modified_no_remove():
    with ft.VarDef([("y", (), "int32", "output", "cpu"),
                    ("z", (), "int32", "inout", "cpu")]) as (y, z):
        y[()] = z[()]
        z[()] = z[()] + 1
        y[()] = y[()] + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("y", (), "int32", "output", "cpu"),
                    ("z", (), "int32", "inout", "cpu")]) as (y, z):
        y[()] = z[()]
        z[()] = z[()] + 1
        y[()] = y[()] + 2
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_reduce_then_reduce():
    with ft.VarDef("y", (), "int32", "inout", "cpu") as y:
        y[()] = y[()] + 1
        y[()] = y[()] + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (), "int32", "inout", "cpu") as y:
        y[()] = y[()] + 3
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_reduce_then_reduce_across_loops():
    with ft.VarDef("y", (4,), "int32", "inout", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] += 1
        with ft.For("i", 0, 4) as i:
            y[i] += 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "inout", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] += 3
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_reduce_then_reduce_across_loops_differnet_indices():
    with ft.VarDef("y", (4,), "int32", "inout", "cpu") as y:
        with ft.For("i", 2, 6) as i:
            y[i - 2] += i
        with ft.For("i", 0, 4) as i:
            y[i] += 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (4,), "int32", "inout", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] += i + 4
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_write_then_multiple_reduces():
    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 1
        y[()] = y[()] + 2
        y[()] = y[()] + 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("y", (), "int32", "output", "cpu") as y:
        y[()] = 6
    std = ft.pop_ast()

    assert std.match(ast)


def test_type1_write_then_loop_then_reduce_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 5) as i:
            y[()] = y[()] + i
        y[()] = y[()] + x[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
        with ft.For("i", 0, 5) as i:
            y[()] = y[()] + i
        y[()] = y[()] + x[()]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_read_by_following_write_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
        y[()] = y[()] * y[()]
        y[()] = y[()] * y[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
        y[()] = y[()] * y[()]
        y[()] = y[()] * y[()]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_later_store():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[()] > 0):
            y[()] = x[()]
        y[()] = 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_later_reduce_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[()] > 0):
            y[()] = x[()]
        y[()] += 1
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[()] > 0):
            y[()] = x[()]
        y[()] += 1
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_earlier_store_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
        with ft.If(x[()] > 0):
            y[()] = x[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 1
        with ft.If(x[()] > 0):
            y[()] = x[()]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type1_not_kill_earlier_reduce_no_remove():
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
        with ft.If(x[()] > 0):
            y[()] += x[()]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] + 1
        with ft.If(x[()] > 0):
            y[()] += x[()]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_type2_inner_loop():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                y[i] = x[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x[i] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_type2_outer_loop():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 8) as j:
                y[j] = x[j] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (8,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("j", 0, 8) as j:
            y[j] = x[j] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_type2_used_no_remove():
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("z", (4,), "int32", "output", "cpu"),
                    ("w", (4,), "int32", "output", "cpu")]) as (x, z, w):
        with ft.VarDef("y", (1,), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 4) as i:
                y[0] = x[i] * 2
                z[i] = y[0] + 1
                w[i] = y[0] + 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("z", (4,), "int32", "output", "cpu"),
                    ("w", (4,), "int32", "output", "cpu")]) as (x, z, w):
        with ft.VarDef("y", (1,), "int32", "cache", "cpu") as y:
            with ft.For("i", 0, 4) as i:
                y[0] = x[i] * 2
                z[i] = y[0] + 1
                w[i] = y[0] + 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_type2_dynamic():
    with ft.VarDef([("n", (), "int32", "input", "byvalue"),
                    ("m", (), "int32", "input", "byvalue")]) as (n, m):
        with ft.Assert(ft.l_and(n[()] > 0, m[()] > 0)):
            with ft.VarDef([
                ("x", (n[()],), "int32", "input", "cpu"),
                ("y", (n[()],), "int32", "output", "cpu"),
            ]) as (x, y):
                with ft.For("i", 0, n[()]) as i:
                    with ft.For("j", 0, m[()]) as j:
                        y[i] = x[i] * 2
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("n", (), "int32", "input", "byvalue"),
                    ("m", (), "int32", "input", "byvalue")]) as (n, m):
        with ft.Assert(ft.l_and(n[()] > 0, m[()] > 0)):
            with ft.VarDef([
                ("x", (n[()],), "int32", "input", "cpu"),
                ("y", (n[()],), "int32", "output", "cpu"),
            ]) as (x, y):
                with ft.For("i", 0, n[()]) as i:
                    y[i] = x[i] * 2
    std = ft.pop_ast()

    assert std.match(ast)


def test_cross_var_def():
    with ft.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, y1, y2):
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = x1[()] + 1
            y1[()] = b[()] * 2
            y2[()] = b[()] * 3
        with ft.VarDef("b", (), "int32", "cache", "cpu") as b:
            b[()] = x2[()] + 2
            y1[()] += b[()] * 2
            y2[()] += b[()] * 3
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("x1", (), "int32", "input", "cpu"),
                    ("x2", (), "int32", "input", "cpu"),
                    ("y1", (), "int32", "output", "cpu"),
                    ("y2", (), "int32", "output", "cpu")]) as (x1, x2, y1, y2):
        with ft.VarDef("b1", (), "int32", "cache", "cpu") as b1:
            b1[()] = x1[()] + 1
            with ft.VarDef("b2", (), "int32", "cache", "cpu") as b2:
                b2[()] = x2[()] + 2
                y1[()] = b1[()] * 2 + b2[()] * 2
                y2[()] = b1[()] * 3 + b2[()] * 3
    std = ft.pop_ast()

    assert std.match(ast)


def test_same_parent_but_dep_and_circular_dependency_on_init():
    with ft.VarDef([("f", (10,), "float32", "output", "cpu"),
                    ("u", (10,), "float32", "cache", "cpu")]) as (f, u):
        with ft.For("l", 0, 10) as l:
            with ft.For("j", 0, 10) as j:
                f[j] = 0
                with ft.For("k", 0, 10) as k:
                    f[j] += u[j]
                    f[j] += 1
                u[j] = f[j]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef([("f", (10,), "float32", "output", "cpu"),
                    ("u", (10,), "float32", "cache", "cpu")]) as (f, u):
        with ft.For("l", 0, 10) as l:
            with ft.For("j", 0, 10) as j:
                f[j] = 0
                with ft.For("k", 0, 10) as k:
                    f[j] += u[j] + 1
                u[j] = f[j]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_circular_dependency_in_parallel():
    with ft.VarDef([("a", (256,), "float32", "inout", "cpu"),
                    ("b", (256,), "float32", "cache", "cpu"),
                    ("c", (256,), "float32", "cache", "cpu")]) as (a, b, c):
        with ft.For("i", 0, 256) as i:
            c[i] = 0
        with ft.For("i", 0, 256) as i:
            b[i] = a[i]
        with ft.For("k", 0, 100) as k:
            ft.MarkNid("L1")
            with ft.For("l", 0, 256) as l:
                c[l] += b[l]
                b[l] = 0
            ft.MarkNid("L")
            with ft.For("i", 0, 256) as i:
                with ft.For("j", 0, 256) as j:
                    b[j] += c[i]
    s = ft.Schedule(ft.pop_ast())
    s.parallelize("L", "openmp")
    ast = s.ast()
    print(ast)
    ast = ft.lower(ast,
                   skip_passes=[
                       "cpu_lower_parallel_reduction", 'scalar_prop_const',
                       'tensor_prop_const', 'prop_one_time_use'
                   ],
                   verbose=1)

    with ft.VarDef([("a", (256,), "float32", "inout", "cpu"),
                    ("c", (256,), "float32", "cache", "cpu")]) as (a, c):
        with ft.For("i", 0, 256) as i:
            c[i] = 0
        with ft.VarDef("b", (256,), "float32", "cache", "cpu") as b:
            with ft.For("i", 0, 256) as i:
                b[i] = a[i]
            with ft.For("k", 0, 100) as k:
                with ft.For("l", 0, 256) as l:
                    c[l] = (c[l] + b[l])
                    b[l] = 0
                with ft.For("i", 0, 256) as i:
                    with ft.For("j", 0, 256) as j:
                        b[j] = (b[j] + c[i])
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)


def test_one_loop_depends_on_multiple_statements_no_remove():
    with ft.VarDef("u", (64,), "float64", "input", "cpu") as u:
        with ft.VarDef("y", (2,), "float64", "output", "cpu") as y:
            with ft.VarDef("tmp", (2,), "float64", "cache", "cpu") as tmp:
                with ft.For("i", 0, 2) as i:
                    tmp[i] = 0
                with ft.VarDef("A", (2,), "float32", "cache", "cpu") as A:
                    A[0] = 0
                    A[1] = 1
                    with ft.For("i", 0, 2) as i:
                        tmp[i] += A[i] * u[0, 0] + A[i] * u[0, 1]
                    A[0] = 2
                    A[1] = 3
                    with ft.For("i", 0, 2) as i:
                        tmp[i] += A[i] * u[1, 0] + A[i] * u[1, 1]
                with ft.For("i", 0, 2) as i:
                    y[i] = tmp[i]
    ast = ft.pop_ast(verbose=True)
    ast = ft.lower(ast,
                   verbose=1,
                   skip_passes=[
                       'scalar_prop_const', 'tensor_prop_const',
                       'prop_one_time_use'
                   ])

    with ft.VarDef("u", (64,), "float64", "input", "cpu") as u:
        with ft.VarDef("y", (2,), "float64", "output", "cpu") as y:
            with ft.VarDef("tmp", (2,), "float64", "cache", "cpu") as tmp:
                with ft.VarDef("A", (2,), "float32", "cache", "cpu") as A:
                    A[0] = 0
                    A[1] = 1
                    with ft.For("i", 0, 2) as i:
                        tmp[i] = A[i] * u[0, 0] + A[i] * u[0, 1]
                    A[0] = 2
                    A[1] = 3
                    with ft.For("i", 0, 2) as i:
                        tmp[i] += A[i] * u[1, 0] + A[i] * u[1, 1]
                with ft.For("i", 0, 2) as i:
                    y[i] = tmp[i]
    std = ft.make_reduction(ft.pop_ast())

    assert std.match(ast)
