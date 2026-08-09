import freetensor as ft


def test_basic_raw_return_list():
    with ft.VarDef("x", (4,), "int32", "cache", "cpu") as x:
        ft.MarkLabel("S0")
        x[0] = 1
        ft.MarkLabel("S1")
        x[1] = x[0]
    ast = ft.pop_ast(verbose=True)

    s0 = ft.find_stmt(ast, "S0")
    s1 = ft.find_stmt(ast, "S1")
    deps = ft.find_deps(ast, dep_type=ft.DEP_RAW)

    assert any(
        d.var == "x" and d.earlier.stmt.id == s0.id and d.later.stmt.id == s1.id
        for d in deps)


def test_exists_and_type_filtering():
    with ft.VarDef("x", (4,), "int32", "cache", "cpu") as x:
        x[0] = 1
        x[1] = x[0]
    ast = ft.pop_ast(verbose=True)

    assert ft.find_deps_exists(ast, dep_type=ft.DEP_RAW)
    assert not ft.find_deps_exists(ast, dep_type=ft.DEP_WAR)


def test_basic_war():
    with ft.VarDef("x", (4,), "int32", "cache", "cpu") as x:
        ft.MarkLabel("S0")
        x[1] = x[0]
        ft.MarkLabel("S1")
        x[0] = 2
    ast = ft.pop_ast(verbose=True)

    s0 = ft.find_stmt(ast, "S0")
    s1 = ft.find_stmt(ast, "S1")
    deps = ft.find_deps(ast, dep_type=ft.DEP_WAR)

    assert any(
        d.var == "x" and d.earlier.stmt.id == s0.id and d.later.stmt.id == s1.id
        for d in deps)


def test_basic_waw():
    with ft.VarDef("x", (4,), "int32", "cache", "cpu") as x:
        ft.MarkLabel("S0")
        x[0] = 1
        ft.MarkLabel("S1")
        x[0] = 2
    ast = ft.pop_ast(verbose=True)

    s0 = ft.find_stmt(ast, "S0")
    s1 = ft.find_stmt(ast, "S1")
    deps = ft.find_deps(ast, dep_type=ft.DEP_WAW)

    assert any(
        d.var == "x" and d.earlier.stmt.id == s0.id and d.later.stmt.id == s1.id
        for d in deps)


def test_direction_normal_and_same():
    with ft.VarDef("x", (8,), "int32", "inout", "cpu") as x:
        with ft.For("i", 1, 8, label="Li") as i:
            x[i] = x[i - 1] + 1
    ast = ft.pop_ast(verbose=True)

    li = ft.find_stmt(ast, "Li")
    assert ft.find_deps_exists(
        ast,
        dep_type=ft.DEP_RAW,
        direction=[[(li, ft.DepDirection.Normal)]],
    )
    assert not ft.find_deps_exists(
        ast,
        dep_type=ft.DEP_RAW,
        direction=[[(li, ft.DepDirection.Same)]],
    )


def test_filter_sub_ast():
    with ft.VarDef([("x", (4,), "int32", "cache", "cpu"),
                    ("y", (4,), "int32", "cache", "cpu")]) as (x, y):
        with ft.For("i", 0, 1, label="Lx"):
            ft.MarkLabel("X0")
            x[0] = 1
            ft.MarkLabel("X1")
            x[1] = x[0]
        with ft.For("i", 0, 1, label="Ly"):
            ft.MarkLabel("Y0")
            y[0] = 1
            ft.MarkLabel("Y1")
            y[1] = y[0]
    ast = ft.pop_ast(verbose=True)

    lx = ft.find_stmt(ast, "Lx")
    deps = ft.find_deps(ast, dep_type=ft.DEP_RAW, filter_sub_ast=lx)

    assert deps
    assert {d.var for d in deps} == {"x"}


def test_ignore_reduction_waw():
    with ft.VarDef("x", (), "int32", "inout", "cpu") as x:
        x[...] += 1
        x[...] += 2
    ast = ft.pop_ast(verbose=True)

    assert not ft.find_deps_exists(
        ast, dep_type=ft.DEP_WAW, ignore_reduction_waw=True)
    assert ft.find_deps_exists(ast,
                               dep_type=ft.DEP_WAW,
                               ignore_reduction_waw=False)


def test_presburger_maps_require_no_project_out_private_axis():
    with ft.VarDef("x", (8,), "int32", "inout", "cpu") as x:
        with ft.For("i", 1, 8, label="Li") as i:
            x[i] = x[i - 1] + 1
    ast = ft.pop_ast(verbose=True)

    li = ft.find_stmt(ast, "Li")
    deps_without_maps = ft.find_deps(
        ast,
        dep_type=ft.DEP_RAW,
        direction=[[(li, ft.DepDirection.Normal)]],
    )
    deps_with_maps = ft.find_deps(
        ast,
        dep_type=ft.DEP_RAW,
        direction=[[(li, ft.DepDirection.Normal)]],
        no_project_out_private_axis=True,
    )

    assert deps_without_maps
    assert deps_with_maps
    assert all(d.later_to_earlier is None for d in deps_without_maps)
    assert any(d.later_to_earlier is not None for d in deps_with_maps)
