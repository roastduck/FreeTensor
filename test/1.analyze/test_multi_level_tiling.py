import freetensor as ft


def test_basic():
    a = 128
    b = 256

    with ft.VarDef([("w", (a, b), "int32", "input", "cpu"),
                    ("x", (b, a), "int32", "input", "cpu"),
                    ("y", (a, a), "int32", "output", "cpu")]) as (
                        w,
                        x,
                        y,
                    ):
        with ft.For("i", 0, a, nid='L1') as i:
            with ft.For("j", 0, a, nid='L2') as j:
                with ft.For("k", 0, b, nid='L3') as k:
                    y[i, j] = y[i, j] + w[i, k] * x[k, j]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L1 S L2 R L3"


def test_basic_2():
    a = 128

    with ft.VarDef([("w", (a, a), "int32", "input", "cpu"),
                    ("x", (a, a), "int32", "input", "cpu"),
                    ("y", (a, a), "int32", "output", "cpu")]) as (
                        w,
                        x,
                        y,
                    ):
        with ft.For("i", 0, a, nid='L1') as i:
            with ft.For("j", 0, a, nid='L2') as j:
                y[i, j] = y[i, j] + w[i, j] * x[0, j]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L1 S L2"


def test_middle_store():
    a = 128
    b = 256
    m = 4

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m), "int32", "output", "cpu")]) as (w, x, y, z):
        with ft.For("i", 0, m, nid='L1') as i:
            with ft.For("j", 0, m, nid='L2') as j:
                z[i, j] = i * j
                with ft.For("p", 0, a, nid='L3') as p:
                    with ft.For("k", 0, b, nid='L4') as k:
                        with ft.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L3 S L5 R L4"


def test_two_branches():
    a = 128
    b = 256
    m = 4

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu")]) as (w, x, y,
                                                                        z):
        with ft.For("i", 0, m, nid='L1') as i:
            with ft.For("j", 0, m, nid='L2') as j:
                with ft.For("k", 0, b, nid='L3') as k:
                    with ft.For("p", 0, a, nid='L4') as p:
                        with ft.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ft.For("p", 0, b, nid='L6') as p:
                    with ft.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = w[i, j, p, 1] * x[i, j, 1, q]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L4 S L5 R L3"
    assert str(fors_with_data_reuse[1]).strip() == "S L6 S L7"


def test_no_data_reuse():
    a = 128
    b = 256
    m = 4

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu")]) as (w, x, y,
                                                                        z):
        with ft.For("i", 0, m, nid='L1') as i:
            with ft.For("j", 0, m, nid='L2') as j:
                with ft.For("k", 0, b, nid='L3') as k:
                    with ft.For("p", 0, a, nid='L4') as p:
                        with ft.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ft.For("p", 0, a, nid='L6') as p:
                    with ft.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L4 S L5 R L3"


def test_no_data_reuse_exchange():
    a = 128
    b = 256
    m = 4

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu")]) as (w, x, y,
                                                                        z):
        with ft.For("i", 0, m, nid='L1') as i:
            with ft.For("j", 0, m, nid='L2') as j:
                with ft.For("p", 0, a, nid='L6') as p:
                    with ft.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
                with ft.For("k", 0, b, nid='L3') as k:
                    with ft.For("p", 0, a, nid='L4') as p:
                        with ft.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L4 S L5 R L3"


def test_root_and_branch():
    a = 128
    b = 256
    m = 4

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu"),
                    ("u", (m, m), "int32", "output", "cpu"),
                    ("v", (m,), "int32", "input", "cpu")]) as (w, x, y, z, u,
                                                               v):
        with ft.For("i", 0, m, nid='L1') as i:
            with ft.For("j", 0, m, nid='L2') as j:
                u[i, j] = j * v[i]
                with ft.For("k", 0, b, nid='L3') as k:
                    with ft.For("p", 0, a, nid='L4') as p:
                        with ft.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ft.For("p", 0, a, nid='L6') as p:
                    with ft.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L4 S L5 R L3"
    assert str(fors_with_data_reuse[1]).strip() == "S L1 S L2"


def test_root_with_no_data_reuse():
    a = 128
    b = 256
    m = 4

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu"),
                    ("u", (m, m), "int32", "output", "cpu")]) as (w, x, y, z,
                                                                  u):
        with ft.For("i", 0, m, nid='L1') as i:
            with ft.For("j", 0, m, nid='L2') as j:
                u[i, j] = i * j
                with ft.For("k", 0, b, nid='L3') as k:
                    with ft.For("p", 0, a, nid='L4') as p:
                        with ft.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ft.For("p", 0, a, nid='L6') as p:
                    with ft.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
    ast = ft.pop_ast()
    fors_with_data_reuse = ft.find_multi_level_tiling(ast)

    assert str(fors_with_data_reuse[0]).strip() == "S L4 S L5 R L3"
