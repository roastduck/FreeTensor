import ir

def test_const_fold():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 0 * i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_partial_fold():
    # This is the case that we need a symbolic bound, instead
    # of using integers only
    with ir.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = 2 * j + i - j - j
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.For("j", 0, 4) as j:
                y[i, j] = i
    std = ir.pop_ast()

    assert std.match(ast)

def test_redundant_if():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 10):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_redundant_if_2():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < i + 2):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_redundant_min():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 10):
                y[i] = ir.min(i, i + 2)
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i
    std = ir.pop_ast()

    assert std.match(ast)

def test_redundant_max():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 10):
                y[i] = ir.max(i, i + 2)
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            y[i] = i + 2
    std = ir.pop_ast()

    assert std.match(ast)

def test_multiple_mins():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.min(ir.min(x[i] + 2, i), x[i])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.min(x[i], i)
    std = ir.pop_ast()

    assert std.match(ast)

def test_multiple_maxes():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.max(ir.max(x[i] + 2, i), x[i])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = ir.max(x[i] + 2, i)
    std = ir.pop_ast()

    assert std.match(ast)

def test_multiple_min_max():
    with ir.VarDef([
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "inout", "cpu")]) as (a, b, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(i < ir.min(ir.max(5, a[()]), ir.max(6, b[()]))):
                y[i] = i
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("a", (), "int32", "input", "cpu"),
            ("b", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "inout", "cpu")]) as (a, b, y):
        with ir.For("i", 0, 4) as i:
            y[i] = i
    std = ir.pop_ast()

    assert std.match(ast)

def test_precondition_from_if():
    with ir.VarDef([
            ("x1", (4,), "int32", "input", "cpu"),
            ("x2", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(x1[i] < x2[i]):
                y[i] = ir.min(x1[i], x2[i])
            with ir.Else():
                y[i] = ir.min(x1[i], x2[i]) + 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x1", (4,), "int32", "input", "cpu"),
            ("x2", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(x1[i] < x2[i]):
                y[i] = x1[i]
            with ir.Else():
                y[i] = x2[i] + 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_multiple_preconditions_from_if():
    with ir.VarDef([
            ("x1", (4,), "int32", "input", "cpu"),
            ("x2", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_and(x1[i] >= 0, x1[i] < x2[i])):
                y[i] = ir.min(x1[i], x2[i])
            with ir.Else():
                y[i] = ir.min(x1[i], x2[i]) + 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x1", (4,), "int32", "input", "cpu"),
            ("x2", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_and(x1[i] >= 0, x1[i] < x2[i])):
                y[i] = x1[i]
            with ir.Else():
                y[i] = ir.min(x1[i], x2[i]) + 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_precondition_from_assert():
    with ir.VarDef([
            ("x1", (4,), "int32", "input", "cpu"),
            ("x2", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.Assert(x1[i] < x2[i]):
                y[i] = ir.min(x1[i], x2[i])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x1", (4,), "int32", "input", "cpu"),
            ("x2", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x1, x2, y):
        with ir.For("i", 0, 4) as i:
            with ir.Assert(x1[i] < x2[i]):
                y[i] = x1[i]
    std = ir.pop_ast()

    assert std.match(ast)

def test_different_scope():
    with ir.VarDef([
            ("x", (4, 10), "int32", "input", "cpu"),
            ("y", (4, 10), "int32", "output", "cpu")]) as (x, y):
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
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4, 10), "int32", "input", "cpu"),
            ("y", (4, 10), "int32", "output", "cpu")]) as (x, y):
        # seperate_tail removes the If nodes
        with ir.For("i", 0, 2) as i:
            with ir.For("j", 0, 5) as j:
                y[i, j] = x[i, j]
        with ir.For("i", 2, 4) as i:
            with ir.For("j", 0, 5) as j:
                y[i, j] = x[i, j] + 2
            with ir.For("j", 5, 10) as j:
                y[i, j] = x[i, j] + 3
    std = ir.pop_ast()

    assert std.match(ast)

def test_dynamic():
    with ir.VarDef([
            ("n", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()]) as i:
            with ir.If(n[()] + 1 > n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("n", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()]) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_floor_div_1():
    with ir.VarDef([
            ("n", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()] // 4) as i:
            with ir.If(i * 4 < n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("n", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, n[()] // 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_floor_div_2():
    with ir.VarDef([
            ("n", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, (n[()] - 1) // 4) as i:
            with ir.If(i * 4 < n[()]):
                y[i] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("n", (), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (n, y):
        with ir.For("i", 0, (n[()] + -1) // 4) as i:
            y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_floor_div_3():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = ir.min(x[()] // 4, x[()] // 4)
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = x[()] // 4
    std = ir.pop_ast()

    assert std.match(ast)

def test_floor_div_4():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = 64 * x[()] // 64
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = x[()]
    std = ir.pop_ast()

    assert std.match(ast)

def test_floor_div_5():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = x[()] // 4 - x[()] // 4
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_mod_1():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = 64 * x[()] % 64
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_mod_2():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        with ir.If(ir.l_and(x[()] >= 0, x[()] < 64)):
            y[()] = x[()] % 64
        with ir.Else():
            y[()] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu")]) as (x, y):
        with ir.If(ir.l_and(x[()] >= 0, x[()] < 64)):
            y[()] = x[()]
        with ir.Else():
            y[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_simplify_not_cmp():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu"),
            ("y3", (4,), "int32", "output", "cpu"),
            ("y4", (4,), "int32", "output", "cpu"),
            ("y5", (4,), "int32", "output", "cpu"),
            ("y6", (4,), "int32", "output", "cpu")]) as (x, y1, y2, y3, y4, y5, y6):
        with ir.For("i", 0, 4) as i:
            y1[i] = ir.l_not(x[i] < 5)
            y2[i] = ir.l_not(x[i] <= 5)
            y3[i] = ir.l_not(x[i] > 5)
            y4[i] = ir.l_not(x[i] >= 5)
            y5[i] = ir.l_not(x[i] == 5)
            y6[i] = ir.l_not(x[i] != 5)
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y1", (4,), "int32", "output", "cpu"),
            ("y2", (4,), "int32", "output", "cpu"),
            ("y3", (4,), "int32", "output", "cpu"),
            ("y4", (4,), "int32", "output", "cpu"),
            ("y5", (4,), "int32", "output", "cpu"),
            ("y6", (4,), "int32", "output", "cpu")]) as (x, y1, y2, y3, y4, y5, y6):
        with ir.For("i", 0, 4) as i:
            y1[i] = x[i] >= 5
            y2[i] = x[i] > 5
            y3[i] = x[i] <= 5
            y4[i] = x[i] < 5
            y5[i] = x[i] != 5
            y6[i] = x[i] == 5
    std = ir.pop_ast()

    assert std.match(ast)

def test_simplify_not_logic_op():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_not(ir.l_and(x[i] >= 0, x[i] < 10))):
                y[i] = x[i]
            with ir.Else():
                y[i] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            with ir.If(ir.l_or(x[i] < 0, x[i] >= 10)):
                y[i] = x[i]
            with ir.Else():
                y[i] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_min_minus_min():
    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu"),
            ("z", (), "int32", "input", "cpu")]) as (x, y, z):
        z[()] = ir.min(x[()], y[()]) - ir.min(x[()], y[()])
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("x", (), "int32", "input", "cpu"),
            ("y", (), "int32", "input", "cpu"),
            ("z", (), "int32", "input", "cpu")]) as (x, y, z):
        z[()] = 0
    std = ir.pop_ast()

    assert std.match(ast)

def test_min_max_as_bound():
    with ir.VarDef([
            ("l", (), "int32", "input", "cpu"),
            ("r", (), "int32", "input", "cpu")]) as (l, r):
        with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ir.For("i", ir.max(l[()], 0), ir.min(r[()], 4)) as i:
                with ir.If(ir.l_and(i >= l[()], i < r[()])):
                    y[i] = 1
                with ir.Else():
                    y[i] = 0
    ast = ir.pop_ast()
    print(ast)
    ast = ir.lower(ast)
    print(ast)

    with ir.VarDef([
            ("l", (), "int32", "input", "cpu"),
            ("r", (), "int32", "input", "cpu")]) as (l, r):
        with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
            with ir.For("i", ir.max(l[()], 0), ir.min(r[()], 4)) as i:
                y[i] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_accessible_after_writing_if():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[0] < 4):
            with ir.If(x[0] < 4):
                y[0] = 1
            x[0] += 1
            with ir.If(x[0] < 4):
                y[0] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.simplify_pass(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[0] < 4):
            y[0] = 1
            x[0] += 1
            with ir.If(x[0] < 4):
                y[0] = 1
    std = ir.pop_ast()

    assert std.match(ast)

def test_accessible_after_writing_for():
    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[0] < 4):
            with ir.For("i", 0, 4) as i:
                with ir.If(x[0] < 4):
                    y[0] = 1
                x[0] += 1
                with ir.If(x[0] < 4):
                    y[0] = 1
    ast = ir.pop_ast()
    print(ast)
    ast = ir.simplify_pass(ast)
    print(ast)

    with ir.VarDef([
            ("x", (4,), "int32", "input", "cpu"),
            ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.If(x[0] < 4):
            with ir.For("i", 0, 4) as i:
                with ir.If(x[0] < 4):
                    y[0] = 1
                x[0] += 1
                with ir.If(x[0] < 4):
                    y[0] = 1
    std = ir.pop_ast()

    assert std.match(ast)
