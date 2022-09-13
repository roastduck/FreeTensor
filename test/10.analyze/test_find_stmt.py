import freetensor as ft


def sorted_ids(stmts):
    ids = [int(s.id) for s in stmts]
    return sorted(ids)


def test_select_child():
    ft.MarkLabel("Vx")
    with ft.VarDef("x", (8,), "float32", "input", "cpu") as x:
        ft.MarkLabel("Vy")
        with ft.VarDef("y", (8,), "float32", "output", "cpu") as y:
            with ft.For("i", 0, 8) as i:
                y[i] = x[i]
    ast = ft.pop_ast(verbose=True)

    results = ft.find_all_stmt(ast, "<VarDef><-<VarDef>")
    results_by_label = ft.find_all_stmt(ast, "Vy")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_child_chained():
    with ft.VarDef("y", (8, 8, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8, label="Li") as i:
            with ft.For("j", 0, 8, label="Lj") as j:
                with ft.For("k", 0, 8, label="Lk") as k:
                    y[i, j, k] = i + j + k
    ast = ft.pop_ast(verbose=True)

    results = ft.find_all_stmt(ast, "<For><-<For><-<For>")
    results_by_label = ft.find_all_stmt(ast, "Lk")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_any_child():
    with ft.VarDef("y", (8, 8, 8), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 8, label="Li") as i:
            with ft.For("j", 0, 8, label="Lj") as j:
                with ft.For("k", 0, 8, label="Lk") as k:
                    ft.MarkLabel("S")
                    y[i, j, k] = i + j + k
    ast = ft.pop_ast(verbose=True)

    results = ft.find_all_stmt(ast, "<-<For>")
    results_by_label = ft.find_all_stmt(ast, "Lk|Lj|S")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_descendant():
    with ft.VarDef("y", (8, 8, 8), "int32", "output", "cpu") as y:
        ft.MarkLabel("S0")
        y[0, 0, 0] = 1
        with ft.For("i", 0, 8, label="Li") as i:
            with ft.For("j", 0, 8, label="Lj") as j:
                with ft.For("k", 0, 8, label="Lk") as k:
                    ft.MarkLabel("S1")
                    y[i, j, k] += i + j + k
    ast = ft.pop_ast(verbose=True)

    results = ft.find_all_stmt(ast, "<ReduceTo><<-<For>")
    results_by_label = ft.find_all_stmt(ast, "S1")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_descendant_with_middle():
    with ft.VarDef([("a", (8, 8), "int32", "input", "cpu"),
                    ("b", (8, 8), "int32", "input", "cpu"),
                    ("c", (8, 8), "int32", "output", "cpu")]) as (a, b, c):
        with ft.For("i", 0, 8, label="Li") as i:
            with ft.For("j", 0, 8, label="Lj") as j:
                c[i, j] = 0
                with ft.For("k", 0, 8, label="Lk") as k:
                    c[i, j] += a[i, k] * b[k, j]
    ast = ft.pop_ast(verbose=True)

    results = ft.find_all_stmt(ast, "<For><-(!<For><-)*Li")
    results_by_label = ft.find_all_stmt(ast, "Lj")
    assert sorted_ids(results) == sorted_ids(results_by_label)

    results = ft.find_all_stmt(ast, "<For><-(!<For><-)*Lj")
    results_by_label = ft.find_all_stmt(ast, "Lk")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_callee_1():

    @ft.inline
    def g(y):
        #! label: L
        for i in range(8):
            y[i] = i

    @ft.transform(verbose=True)
    def f(y: ft.Var[(8, 8), "int32", "output"]):
        #! label: L
        for i in range(8):
            #! label: g
            g(y[i])

    results = ft.find_all_stmt(f, "L<~g")
    results_by_tree = ft.find_all_stmt(f, "L<-L")
    assert sorted_ids(results) == sorted_ids(results_by_tree)


def test_select_callee_2():

    @ft.inline
    def g(y):
        #! label: L2
        for i in range(8):
            y[i] = i

    @ft.transform(verbose=True)
    def f(y: ft.Var[(8, 8), "int32", "output"]):
        #! label: L1
        for i in range(8):
            #! label: g
            g(y[i])

    results = ft.find_all_stmt(f, "<For><~g")
    results_by_label = ft.find_all_stmt(f, "L2")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_indirect_callee():

    @ft.inline
    def h(y):
        #! label: L3
        for i in range(8):
            y[i] = i

    @ft.inline
    def g(y):
        #! label: L2
        for i in range(8):
            #! label: h
            h(y[i])

    @ft.transform(verbose=True)
    def f(y: ft.Var[(8, 8, 8), "int32", "output"]):
        #! label: L1
        for i in range(8):
            #! label: g
            g(y[i])

    results = ft.find_all_stmt(f, "<For><<~g")
    results_by_label = ft.find_all_stmt(f, "L2|L3")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_select_indirect_callee_anonymous_call_site():

    @ft.inline
    def h(y):
        #! label: L3
        for i in range(8):
            y[i] = i

    @ft.inline
    def g(y):
        #! label: L2
        for i in range(8):
            # Anonymous call site!
            h(y[i])

    @ft.transform(verbose=True)
    def f(y: ft.Var[(8, 8, 8), "int32", "output"]):
        #! label: L1
        for i in range(8):
            #! label: g
            g(y[i])

    results = ft.find_all_stmt(f, "<For><<~g")
    results_by_label = ft.find_all_stmt(f, "L2|L3")
    assert sorted_ids(results) == sorted_ids(results_by_label)


def test_find_in_dfs_pre_order():
    with ft.VarDef([("y", (8, 8), "int32", "output", "cpu"),
                    ("z", (8, 8), "int32", "output", "cpu")]) as (y, z):
        with ft.For("i", 0, 8, label="Li") as i:
            with ft.For("j", 0, 8, label="Lj") as j:
                ft.MarkLabel("S0")
                y[i, j] = i + j
                ft.MarkLabel("S1")
                z[i, j] = i * j
    ast = ft.pop_ast(verbose=True)

    results = ft.find_all_stmt(ast, "(<For>|<Store>)<<-Li")
    results_by_label = [
        ft.find_stmt(ast, "Lj"),
        ft.find_stmt(ast, "S0"),
        ft.find_stmt(ast, "S1")
    ]
    assert results == results_by_label  # No sort
