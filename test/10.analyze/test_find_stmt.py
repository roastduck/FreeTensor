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
