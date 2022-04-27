import ir
import pytest

# This is a common test for pass/simplify and pass/z3_simplify.
# Some tests are supported by both passes, and some tests are
# supported by one of them. We use @pytest.mark.parametrize to
# test these two passes


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_const_fold(p):
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 0 * i
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_partial_fold(p):
    # This is the case that we need a symbolic bound, instead
    # of using integers only
    with ir.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = 2 * j + i - j - j
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = i
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_redundant_if(p):
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 10):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_redundant_if_2(p):
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < i + 2):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_redundant_if_3(p):
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()]) as i:
            with ir.If(2 * i < i + n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()]) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_redundant_min(p):
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 10):
                y[i] = ir.min(i, i + 2)
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_redundant_max(p):
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 10):
                y[i] = ir.max(i, i + 2)
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i + 2
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_multiple_mins_1(p):
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.min(ir.min(x[i] + 2, i), x[i])
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.min(x[i], i)
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_multiple_mins_2(p):
    with ir.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10) as i:
            with ir.For("j", 0, 10) as j:
                with ir.For("k", 0, 10) as k:
                    y[i, j, k] = ir.max(i + j - k, ir.max(i - k, i + j + -1))
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10) as i:
            with ir.For("j", 0, 10) as j:
                with ir.For("k", 0, 10) as k:
                    y[i, j, k] = ir.max(i - k, i + j + -1)
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_multiple_maxes_1(p):
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.max(ir.max(x[i] + 2, i), x[i])
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.max(x[i] + 2, i)
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_multiple_maxes_2(p):
    with ir.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10) as i:
            with ir.For("j", 0, 10) as j:
                with ir.For("k", 0, 10) as k:
                    y[i, j, k] = ir.max(i + j - k, ir.max(i - k, i + j + -1))
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("y", (10, 10, 10), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 10) as i:
            with ir.For("j", 0, 10) as j:
                with ir.For("k", 0, 10) as k:
                    y[i, j, k] = ir.max(i + j - k, i + j + -1)
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_multiple_min_max(p):
    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (a, b, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i < ir.min(ir.max(5, a[()]), ir.max(6, b[()]))):
                y[i] = i
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("a", (), "int32", "input", "cpu"),
        ("b", (), "int32", "input", "cpu"),
        ("y", (4,), "int32", "inout", "cpu"),
    ]) as (a, b, y):
        with ir.For("i", 0, 4) as i:
            y[i] = i
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_precondition_from_if(p):
    with ir.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(x1[i] < x2[i]):
                y[i] = ir.min(x1[i], x2[i])
            with ir.Else():
                y[i] = ir.min(x1[i], x2[i]) + 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(x1[i] < x2[i]):
                y[i] = x1[i]
            with ir.Else():
                y[i] = x2[i] + 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_multiple_preconditions_from_if(p):
    with ir.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_and(x1[i] >= 0, x1[i] < x2[i])):
                y[i] = ir.min(x1[i], x2[i])
            with ir.Else():
                y[i] = ir.min(x1[i], x2[i]) + 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_and(x1[i] >= 0, x1[i] < x2[i])):
                y[i] = x1[i]
            with ir.Else():
                y[i] = ir.min(x1[i], x2[i]) + 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_precondition_from_assert(p):
    with ir.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.Assert(x1[i] < x2[i]):
                y[i] = ir.min(x1[i], x2[i])
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("x1", (4,), "int32", "input", "cpu"),
        ("x2", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.Assert(x1[i] < x2[i]):
                y[i] = x1[i]
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_assert_false(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[()] > 0):
            with ir.Assert(x[()] < 0):
                y[()] = 1
    ast = ir.pop_ast()
    print(ast)
    with pytest.raises(ir.AssertAlwaysFalse):
        ast = p(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_unreachable_assert_false(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[()] < 0):
            with ir.If(x[()] > 0):
                with ir.Assert(x[()] > 0):
                    y[()] = 1
            with ir.Else():
                y[()] = 2
        with ir.Else():
            y[()] = 3
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[()] < 0):
            y[()] = 2
        with ir.Else():
            y[()] = 3
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_different_scope(p):
    with ir.VarDef([
        ("x", (4, 10), "int32", "input", "cpu"),
        ("y", (4, 10), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 2):
                with ir.For("j", 0, 5) as j:
                    with ir.If(j < 5):
                        y[i, j] = x[i, j]
                    with ir.Else():
                        y[i, j] = x[i, j] + 2
            with ir.Else():
                with ir.For("j", 0, 10) as j:
                    with ir.If(j < 5):
                        y[i, j] = x[i, j] + 2
                    with ir.Else():
                        y[i, j] = x[i, j] + 3
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4, 10), "int32", "input", "cpu"),
        ("y", (4, 10), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 2):
                with ir.For("j", 0, 5) as j:
                    y[i, j] = x[i, j]
            with ir.Else():
                with ir.For("j", 0, 10) as j:
                    with ir.If(j < 5):
                        y[i, j] = x[i, j] + 2
                    with ir.Else():
                        y[i, j] = x[i, j] + 3
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_dynamic(p):
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()]) as i:
            with ir.If(n[()] + 1 > n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()]) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_floor_div_1(p):
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()] // 4) as i:
            with ir.If(i * 4 < n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()] // 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_floor_div_2(p):
    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, (n[()] - 1) // 4) as i:
            with ir.If(i * 4 < n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("n", (), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, (n[()] - 1) // 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_floor_div_3(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ir.min(x[()] // 4, x[()] // 4)
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] // 4
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_floor_div_4(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 64 * x[()] // 64
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()]
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_floor_div_5(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] // 4 - x[()] // 4
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_floor_div_6(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] // -1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * -1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_mod_1(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 64 * x[()] % 64
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_mod_2(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(ir.l_and(x[()] >= 0, x[()] < 64)):
            y[()] = x[()] % 64
        with ir.Else():
            y[()] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        with ir.If(ir.l_and(x[()] >= 0, x[()] < 64)):
            y[()] = x[()]
        with ir.Else():
            y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_simplify_not_cmp(p):
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
        ("y5", (4,), "int32", "output", "cpu"),
        ("y6", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2, y3, y4, y5, y6):
        with ir.For("i", 0, 4) as i:
            y1[i] = ir.l_not(x[i] < 5)
            y2[i] = ir.l_not(x[i] <= 5)
            y3[i] = ir.l_not(x[i] > 5)
            y4[i] = ir.l_not(x[i] >= 5)
            y5[i] = ir.l_not(x[i] == 5)
            y6[i] = ir.l_not(x[i] != 5)
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
        ("y3", (4,), "int32", "output", "cpu"),
        ("y4", (4,), "int32", "output", "cpu"),
        ("y5", (4,), "int32", "output", "cpu"),
        ("y6", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2, y3, y4, y5, y6):
        with ir.For("i", 0, 4) as i:
            y1[i] = x[i] >= 5
            y2[i] = x[i] > 5
            y3[i] = x[i] <= 5
            y4[i] = x[i] < 5
            y5[i] = x[i] != 5
            y6[i] = x[i] == 5
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_simplify_not_logic_op(p):
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_not(ir.l_and(x[i] >= 0, x[i] < 10))):
                y[i] = x[i]
            with ir.Else():
                y[i] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_or(x[i] < 0, x[i] >= 10)):
                y[i] = x[i]
            with ir.Else():
                y[i] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_min_minus_min(p):
    with ir.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y", (), "int32", "input", "cpu"),
        ("z", (), "int32", "input", "cpu"),
    ]) as (x, y, z):
        z[()] = ir.min(x[()], y[()]) - ir.min(x[()], y[()])
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([
        ("x", (), "int32", "input", "cpu"),
        ("y", (), "int32", "input", "cpu"),
        ("z", (), "int32", "input", "cpu"),
    ]) as (x, y, z):
        z[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_min_max_as_bound(p):
    with ir.VarDef([("l", (), "int32", "input", "cpu"),
                    ("r", (), "int32", "input", "cpu")]) as (l, r):
        with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ir.For("i", ir.max(l[()], 0), ir.min(r[()], 4)) as i:
                with ir.If(ir.l_and(i >= l[()], i < r[()])):
                    y[i] = 1
                with ir.Else():
                    y[i] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("l", (), "int32", "input", "cpu"),
                    ("r", (), "int32", "input", "cpu")]) as (l, r):
        with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ir.For("i", ir.max(l[()], 0), ir.min(r[()], 4)) as i:
                y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_accessible_after_writing_if(p):
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[0] < 4):
            with ir.If(x[0] < 4):
                y[0] = 1
            x[0] += 1
            with ir.If(x[0] < 4):
                y[1] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("x", (4,), "int32", "input", "cpu") as x:
        with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ir.If(x[0] < 4):
                y[0] = 1
                x[0] += 1
                with ir.If(x[0] < 4):
                    y[1] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass, ir.z3_simplify])
def test_accessible_after_writing_for(p):
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[0] < 4):
            with ir.For("i", 0, 4) as i:
                with ir.If(x[0] < 4):
                    y[0] = 1
                x[0] += 1
                with ir.If(x[0] < 4):
                    y[1] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("x", (4,), "int32", "input", "cpu") as x:
        with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ir.If(x[0] < 4):
                with ir.For("i", 0, 4) as i:
                    with ir.If(x[0] < 4):
                        y[0] = 1
                    x[0] += 1
                    with ir.If(x[0] < 4):
                        y[1] = 1
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.simplify_pass])
def test_loop_length_0_or_1(p):
    with ir.VarDef("n", (), "int32", "input", "cpu") as n:
        with ir.Assert(n[()] <= 1):
            with ir.VarDef("y", (n[()],), "int32", "inout", "cpu") as y:
                with ir.For("i", 0, n[()]) as i:
                    y[i] = i
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef("n", (), "int32", "input", "cpu") as n:
        with ir.Assert(n[()] <= 1):
            with ir.VarDef("y", (n[()],), "int32", "inout", "cpu") as y:
                with ir.If(n[()] == 1):
                    y[0] = 0
    std = ir.pop_ast()

    assert std.match(ast)


@pytest.mark.parametrize('p', [ir.z3_simplify])
def test_complex_tautology(p):
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("pred", (), "bool", "input", "cpu"),
                    ("y", (), "bool", "output", "cpu")]) as (x, pred, y):
        y[()] = ir.l_and(ir.l_or(x[()] < 5, ir.l_and(x[()] >= 5, True)),
                         pred[()])
    ast = ir.pop_ast()
    print(ast)
    ast = p(ast)
    print(ast)

    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("pred", (), "bool", "input", "cpu"),
                    ("y", (), "bool", "output", "cpu")]) as (x, pred, y):
        y[()] = pred[()]
    std = ir.pop_ast()

    assert std.match(ast)
