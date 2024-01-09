import freetensor as ft
import pytest

# This is a common test for pass/simplify and pass/z3_simplify.
# Some tests are supported by both passes, and some tests are
# supported by one of them. We use @pytest.mark.parametrize to
# test these two passes


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_const_fold(p):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 0 * i
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_partial_fold(p):
    # This is the case that we need a symbolic bound, instead
    # of using integers only
    with ft.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = 2 * j + i - j - j
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.For("j", 0, 4) as j:
                y[i, j] = i
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_redundant_if(p):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 10):
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_redundant_if_2(p):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.If(i < i + 2):
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_redundant_if_3(p):
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()]) as i:
            with ft.If(2 * i < i + n[()]):
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()]) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_int_max(p):
    with ft.VarDef([("a", (20, 64), "int32", "input", "cpu"),
                    ("b", (20, 64), "int32", "output", "cpu")]) as (a, b):
        with ft.For("i", 0, 20) as i:
            with ft.For("j", 0, 2147483647) as j:
                with ft.If(j < ft.min(-32 * (i % 4) + 100, 64)):
                    b[i, j] = a[i, j] + 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("a", (20, 64), "int32", "input", "cpu"),
                    ("b", (20, 64), "int32", "output", "cpu")]) as (a, b):
        with ft.For("i", 0, 20) as i:
            with ft.For("j", 0, 2147483647) as j:
                with ft.If(j < ft.min(-32 * (i % 4) + 100, 64)):
                    b[i, j] = a[i, j] + 1
    std = ft.pop_ast()

    assert std.match(ast)  # Unchanged


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_redundant_min(p):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 10):
                y[i] = ft.min(i, i + 2)
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_redundant_max(p):
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 10):
                y[i] = ft.max(i, i + 2)
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = i + 2
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_multiple_mins_1(p):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.min(ft.min(x[i] + 2, i), x[i])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.min(x[i], i)
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.simplify])
def test_multiple_mins_2(p):
    with ft.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10) as i:
            with ft.For("j", 0, 10) as j:
                with ft.For("k", 0, 10) as k:
                    y[i, j, k] = ft.min(i + j - k, ft.min(i - k, i + j + -1))
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10) as i:
            with ft.For("j", 0, 10) as j:
                with ft.For("k", 0, 10) as k:
                    y[i, j, k] = ft.min(i + -1 * k, i + j + -1)
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_multiple_maxes_1(p):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.max(ft.max(x[i] + 2, i), x[i])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.max(x[i] + 2, i)
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.simplify])
def test_multiple_maxes_2(p):
    with ft.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10) as i:
            with ft.For("j", 0, 10) as j:
                with ft.For("k", 0, 10) as k:
                    y[i, j, k] = ft.max(i + j - k, ft.max(i - k, i + j + -1))
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 10) as i:
            with ft.For("j", 0, 10) as j:
                with ft.For("k", 0, 10) as k:
                    y[i, j, k] = ft.max(i + j + -1 * k, i + j + -1)
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.simplify, ft.z3_simplify])
def test_multiple_min_max(p):
    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (a, b, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(i < ft.min(ft.max(5, a[()]), ft.max(6, b[()]))):
                y[i] = i
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (a, b, y):
        with ft.For("i", 0, 4) as i:
            y[i] = i
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.simplify])
def test_multiple_mins_separted_by_scalar_op(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[...] = ft.min(10 * ft.min(x[...], 8), 50)
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[...] = ft.min(10 * x[...], 50)
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_precondition_from_if(p):
    with ft.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(x1[i] < x2[i]):
                y[i] = ft.min(x1[i], x2[i])
            with ft.Else():
                y[i] = ft.min(x1[i], x2[i]) + 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(x1[i] < x2[i]):
                y[i] = x1[i]
            with ft.Else():
                y[i] = x2[i] + 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_multiple_preconditions_from_if(p):
    with ft.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_and(x1[i] >= 0, x1[i] < x2[i])):
                y[i] = ft.min(x1[i], x2[i])
            with ft.Else():
                y[i] = ft.min(x1[i] + 1, x2[i] + 1)
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_and(x1[i] >= 0, x1[i] < x2[i])):
                y[i] = x1[i]
            with ft.Else():
                y[i] = ft.min(x1[i] + 1, x2[i] + 1)
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_precondition_from_assert(p):
    with ft.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.Assert(x1[i] < x2[i]):
                y[i] = ft.min(x1[i], x2[i])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            with ft.Assert(x1[i] < x2[i]):
                y[i] = x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_assert_false(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[()] > 0):
            with ft.Assert(x[()] < 0):
                y[()] = 1
    ast = ft.pop_ast(verbose=True)
    with pytest.raises(ft.AssertAlwaysFalse):
        ast = p(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_unreachable_assert_false(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[()] < 0):
            with ft.If(x[()] > 0):
                with ft.Assert(x[()] > 0):
                    y[()] = 1
            with ft.Else():
                y[()] = 2
        with ft.Else():
            y[()] = 3
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[()] < 0):
            y[()] = 2
        with ft.Else():
            y[()] = 3
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.simplify, ft.z3_simplify])
def test_precondition_from_sign_type(p):
    with ft.VarDef([
        ("x1", (4,), "int32<0", "input", "cpu"),
        ("x2", (4,), "int32>0", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            y[i] = ft.min(x1[i], x2[i])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x1", (4,), "int32<0", "input", "cpu"),
        ("x2", (4,), "int32>0", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ft.For("i", 0, 4) as i:
            y[i] = x1[i]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_different_scope(p):
    with ft.VarDef([
        ("x", (4, 10), "int32", "input", "cpu"),
        ("y", (4, 10), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 2):
                with ft.For("j", 0, 5) as j:
                    with ft.If(j < 5):
                        y[i, j] = x[i, j]
                    with ft.Else():
                        y[i, j] = x[i, j] + 2
            with ft.Else():
                with ft.For("j", 0, 10) as j:
                    with ft.If(j < 5):
                        y[i, j] = x[i, j] + 2
                    with ft.Else():
                        y[i, j] = x[i, j] + 3
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x", (4, 10), "int32", "input", "cpu"),
        ("y", (4, 10), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(i < 2):
                with ft.For("j", 0, 5) as j:
                    y[i, j] = x[i, j]
            with ft.Else():
                with ft.For("j", 0, 10) as j:
                    with ft.If(j < 5):
                        y[i, j] = x[i, j] + 2
                    with ft.Else():
                        y[i, j] = x[i, j] + 3
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_dynamic(p):
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()]) as i:
            with ft.If(n[()] + 1 > n[()]):
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()]) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_floor_div_1(p):
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()] // 4) as i:
            with ft.If(i * 4 < n[()]):
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, n[()] // 4) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_floor_div_2(p):
    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, (n[()] - 1) // 4) as i:
            with ft.If(i * 4 < n[()]):
                y[i] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ft.For("i", 0, (n[()] - 1) // 4) as i:
            y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_floor_div_3(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ft.min(x[()] // 4, x[()] // 4)
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] // 4
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_floor_div_4(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 64 * x[()] // 64
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_floor_div_5(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] // 4 - x[()] // 4
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_floor_div_6(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] // -1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * -1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_mod_1(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 64 * x[()] % 64
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_mod_2(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(ft.l_and(x[()] >= 0, x[()] < 64)):
            y[()] = x[()] % 64
        with ft.Else():
            y[()] = 0
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ft.If(ft.l_and(x[()] >= 0, x[()] < 64)):
            y[()] = x[()]
        with ft.Else():
            y[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_divisible_div(p):
    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "output", "cpu")]) as (a, b, c):
        c[...] = a[...] * b[...] // b[...]
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "output", "cpu")]) as (a, b, c):
        c[...] = a[...]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_divisible_mod(p):
    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "output", "cpu")]) as (a, b, c):
        c[...] = a[...] * b[...] % b[...]
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "output", "cpu")]) as (a, b, c):
        c[...] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_reduce_fraction_for_div(p):
    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "input", "cpu"),
                    ("d", (), "int32", "output", "cpu")]) as (a, b, c, d):
        d[...] = (a[...] * b[...]) // (b[...] * c[...])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "input", "cpu"),
                    ("d", (), "int32", "output", "cpu")]) as (a, b, c, d):
        d[...] = a[...] // c[...]
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_not_reduce_fraction_for_mod(p):
    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "input", "cpu"),
                    ("d", (), "int32", "output", "cpu")]) as (a, b, c, d):
        d[...] = (a[...] * b[...]) % (b[...] * c[...])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("a", (), "int32", "input", "cpu"),
                    ("b", (), "int32", "input", "cpu"),
                    ("c", (), "int32", "input", "cpu"),
                    ("d", (), "int32", "output", "cpu")]) as (a, b, c, d):
        d[...] = (a[...] * b[...]) % (b[...] * c[...])
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_simplify_not_cmp(p):
    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "bool", "output", "cpu"),
        ("y2", (4,), "bool", "output", "cpu"),
        ("y3", (4,), "bool", "output", "cpu"),
        ("y4", (4,), "bool", "output", "cpu"),
        ("y5", (4,), "bool", "output", "cpu"),
        ("y6", (4,), "bool", "output", "cpu"),
    ]) as (x, y1, y2, y3, y4, y5, y6):
        with ft.For("i", 0, 4) as i:
            y1[i] = ft.l_not(x[i] < 5)
            y2[i] = ft.l_not(x[i] <= 5)
            y3[i] = ft.l_not(x[i] > 5)
            y4[i] = ft.l_not(x[i] >= 5)
            y5[i] = ft.l_not(x[i] == 5)
            y6[i] = ft.l_not(x[i] != 5)
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "bool", "output", "cpu"),
        ("y2", (4,), "bool", "output", "cpu"),
        ("y3", (4,), "bool", "output", "cpu"),
        ("y4", (4,), "bool", "output", "cpu"),
        ("y5", (4,), "bool", "output", "cpu"),
        ("y6", (4,), "bool", "output", "cpu"),
    ]) as (x, y1, y2, y3, y4, y5, y6):
        with ft.For("i", 0, 4) as i:
            y1[i] = x[i] >= 5
            y2[i] = x[i] > 5
            y3[i] = x[i] <= 5
            y4[i] = x[i] < 5
            y5[i] = x[i] != 5
            y6[i] = x[i] == 5
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_simplify_not_logic_op(p):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_not(ft.l_and(x[i] >= 0, x[i] < 10))):
                y[i] = x[i]
            with ft.Else():
                y[i] = 0
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_or(x[i] < 0, x[i] >= 10)):
                y[i] = x[i]
            with ft.Else():
                y[i] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_simplify_identical_term_in_logic_or(p):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_or(x[i] < 0, ft.l_or(x[i] >= 10, x[i] < 0))):
                y[i] = x[i]
            with ft.Else():
                y[i] = 0
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_or(x[i] < 0, x[i] >= 10)):
                y[i] = x[i]
            with ft.Else():
                y[i] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_simplify_identical_term_in_logic_and(p):
    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_and(x[i] >= 0, ft.l_and(x[i] < 10, x[i] >= 0))):
                y[i] = x[i]
            with ft.Else():
                y[i] = 0
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.For("i", 0, 4) as i:
            with ft.If(ft.l_and(x[i] >= 0, x[i] < 10)):
                y[i] = x[i]
            with ft.Else():
                y[i] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_min_minus_min(p):
    with ft.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y", (), "int32", "input", "cpu"),
        ("z", (), "int32", "output", "cpu"),
    ]) as (x, y, z):
        z[()] = ft.min(x[()], y[()]) - ft.min(x[()], y[()])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y", (), "int32", "input", "cpu"),
        ("z", (), "int32", "output", "cpu"),
    ]) as (x, y, z):
        z[()] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_min_max_as_bound(p):
    with ft.VarDef([("l", (), "int32", "input", "cpu"),
                    ("r", (), "int32", "input", "cpu")]) as (l, r):
        with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ft.For("i", ft.max(l[()], 0), ft.min(r[()], 4)) as i:
                with ft.If(ft.l_and(i >= l[()], i < r[()])):
                    y[i] = 1
                with ft.Else():
                    y[i] = 0
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("l", (), "int32", "input", "cpu"),
                    ("r", (), "int32", "input", "cpu")]) as (l, r):
        with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ft.For("i", ft.max(l[()], 0), ft.min(r[()], 4)) as i:
                y[i] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_accessible_after_writing_if(p):
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[0] < 4):
            with ft.If(x[0] < 4):
                y[0] = 1
            x[0] += 1
            with ft.If(x[0] < 4):
                y[1] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("x", (4,), "int32", "inout", "cpu") as x:
        with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ft.If(x[0] < 4):
                y[0] = 1
                x[0] += 1
                with ft.If(x[0] < 4):
                    y[1] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify, ft.z3_simplify])
def test_accessible_after_writing_for(p):
    with ft.VarDef([("x", (4,), "int32", "inout", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ft.If(x[0] < 4):
            with ft.For("i", 0, 4) as i:
                with ft.If(x[0] < 4):
                    y[0] = 1
                x[0] += 1
                with ft.If(x[0] < 4):
                    y[1] = 1
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("x", (4,), "int32", "inout", "cpu") as x:
        with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ft.If(x[0] < 4):
                with ft.For("i", 0, 4) as i:
                    with ft.If(x[0] < 4):
                        y[0] = 1
                    x[0] += 1
                    with ft.If(x[0] < 4):
                        y[1] = 1
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.pb_simplify, ft.simplify])
def test_loop_length_0_or_1(p):
    with ft.VarDef("n", (), "int32", "input", "cpu") as n:
        with ft.Assert(n[()] <= 1):
            with ft.VarDef("y", (n[()],), "int32", "inout", "cpu") as y:
                with ft.For("i", 0, n[()]) as i:
                    y[i] = i
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef("n", (), "int32", "input", "cpu") as n:
        with ft.Assert(n[()] <= 1):
            with ft.VarDef("y", (n[()],), "int32", "inout", "cpu") as y:
                with ft.If(n[()] == 1):
                    y[0] = 0
    std = ft.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ft.z3_simplify])
def test_complex_tautology(p):
    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("pred", (), "bool", "input", "cpu"),
                    ("y", (), "bool", "output", "cpu")]) as (x, pred, y):
        y[()] = ft.l_and(ft.l_or(x[()] < 5, ft.l_and(x[()] >= 5, True)),
                         pred[()])
    ast = ft.pop_ast(verbose=True)
    ast = p(ast)
    print(ast)

    with ft.VarDef([("x", (), "int32", "input", "cpu"),
                    ("pred", (), "bool", "input", "cpu"),
                    ("y", (), "bool", "output", "cpu")]) as (x, pred, y):
        y[()] = pred[()]
    std = ft.pop_ast()

    assert std.match(ast)
