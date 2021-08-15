import ir


def test_basic():
    a = 128
    b = 256

    with ir.VarDef([("w", (a, b), "int32", "input", "cpu"),
                    ("x", (b, a), "int32", "input", "cpu"),
                    ("y", (a, a), "int32", "output", "cpu")]) as (
                        w,
                        x,
                        y,
                    ):
        with ir.For("i", 0, a, nid='L1') as i:
            with ir.For("j", 0, a, nid='L2') as j:
                with ir.For("k", 0, b, nid='L3') as k:
                    y[i, j] = y[i, j] + w[i, k] * x[k, j]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L2 S L1 R L3"


def test_basic_2():
    a = 128

    with ir.VarDef([("w", (a, a), "int32", "input", "cpu"),
                    ("x", (a, a), "int32", "input", "cpu"),
                    ("y", (a, a), "int32", "output", "cpu")]) as (
                        w,
                        x,
                        y,
                    ):
        with ir.For("i", 0, a, nid='L1') as i:
            with ir.For("j", 0, a, nid='L2') as j:
                y[i, j] = y[i, j] + w[i, j] * x[0, j]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L2 S L1"


def test_middle_store():
    a = 128
    b = 256
    m = 4

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m), "int32", "output", "cpu")]) as (w, x, y, z):
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                z[i, j] = i * j
                with ir.For("p", 0, a, nid='L3') as p:
                    with ir.For("k", 0, b, nid='L4') as k:
                        with ir.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L5 S L3 R L4"


def test_two_branches():
    a = 128
    b = 256
    m = 4

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu")]) as (w, x, y,
                                                                        z):
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                with ir.For("k", 0, b, nid='L3') as k:
                    with ir.For("p", 0, a, nid='L4') as p:
                        with ir.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ir.For("p", 0, b, nid='L6') as p:
                    with ir.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = w[i, j, p, 1] * x[i, j, 1, q]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L5 S L4 R L3"
    assert fors_with_data_reuse[1].strip() == "S L7 S L6"


def test_no_data_reuse():
    a = 128
    b = 256
    m = 4

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu")]) as (w, x, y,
                                                                        z):
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                with ir.For("k", 0, b, nid='L3') as k:
                    with ir.For("p", 0, a, nid='L4') as p:
                        with ir.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ir.For("p", 0, a, nid='L6') as p:
                    with ir.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L5 S L4 R L3"


def test_no_data_reuse_exchange():
    a = 128
    b = 256
    m = 4

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu")]) as (w, x, y,
                                                                        z):
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                with ir.For("p", 0, a, nid='L6') as p:
                    with ir.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
                with ir.For("k", 0, b, nid='L3') as k:
                    with ir.For("p", 0, a, nid='L4') as p:
                        with ir.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L5 S L4 R L3"


def test_root_and_branch():
    a = 128
    b = 256
    m = 4

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu"),
                    ("u", (m, m), "int32", "output", "cpu"),
                    ("v", (m,), "int32", "input", "cpu")]) as (w, x, y, z, u,
                                                               v):
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                u[i, j] = j * v[i]
                with ir.For("k", 0, b, nid='L3') as k:
                    with ir.For("p", 0, a, nid='L4') as p:
                        with ir.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ir.For("p", 0, a, nid='L6') as p:
                    with ir.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L5 S L4 R L3"
    assert fors_with_data_reuse[1].strip() == "S L2 S L1"


def test_root_with_no_data_reuse():
    a = 128
    b = 256
    m = 4

    with ir.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("z", (m, m, a, a), "int32", "output", "cpu"),
                    ("u", (m, m), "int32", "output", "cpu")]) as (w, x, y, z,
                                                                  u):
        with ir.For("i", 0, m, nid='L1') as i:
            with ir.For("j", 0, m, nid='L2') as j:
                u[i, j] = i * j
                with ir.For("k", 0, b, nid='L3') as k:
                    with ir.For("p", 0, a, nid='L4') as p:
                        with ir.For("q", 0, a, nid='L5') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]
                with ir.For("p", 0, a, nid='L6') as p:
                    with ir.For("q", 0, a, nid='L7') as q:
                        z[i, j, p, q] = y[i, j, p, q]
    ast = ir.pop_ast()
    fors_with_data_reuse = ir.find_multi_level_tiling(ast)

    assert fors_with_data_reuse[0].strip() == "S L5 S L4 R L3"
