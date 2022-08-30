import freetensor as ft


def test_flop():
    with ft.VarDef([
        ("x1", (32,), "float32", "input", "cpu"),
        ("x2", (32,), "float32", "input", "cpu"),
        ("x3", (32,), "float32", "input", "cpu"),
        ("y", (32,), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        with ft.For("i", 0, 32, label='L1') as i:
            ft.MarkLabel('S1')
            y[i] = x1[i] * x2[i] + x3[i]
    ast = ft.pop_ast(verbose=True)

    features = ft.structural_feature(ast)

    S1 = features[ft.lookup_id(ast, 'S1')]
    L1 = features[ft.lookup_id(ast, 'L1')]

    assert S1.op_cnt[ft.DataType('float32')] == 2
    assert L1.op_cnt[ft.DataType('float32')] == 64


def test_access_count():
    with ft.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y", (32,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 32, label='L1') as i:
            ft.MarkLabel('S1')
            y[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)

    features = ft.structural_feature(ast)

    S1 = features[ft.lookup_id(ast, 'S1')]
    L1 = features[ft.lookup_id(ast, 'L1')]

    assert S1.load_cnt[ft.MemType('cpu')] == 1
    assert S1.store_cnt[ft.MemType('cpu')] == 1
    assert S1.access_cnt[ft.MemType('cpu')] == 2
    assert L1.load_cnt[ft.MemType('cpu')] == 32
    assert L1.store_cnt[ft.MemType('cpu')] == 32
    assert L1.access_cnt[ft.MemType('cpu')] == 64


def test_access_count_overlap():
    with ft.VarDef([
        ("x", (48,), "int32", "input", "cpu"),
        ("y1", (48,), "int32", "output", "cpu"),
        ("y2", (48,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.NamedScope('S1'):
            with ft.For("i", 0, 32, label='L1') as i:
                y1[i] = x[i] + 1
            with ft.For("i", 16, 48, label='L2') as i:
                y2[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)

    features = ft.structural_feature(ast)

    S1 = features[ft.lookup_id(ast, 'S1')]
    L1 = features[ft.lookup_id(ast, 'L1')]
    L2 = features[ft.lookup_id(ast, 'L2')]

    assert L1.load_cnt[ft.MemType('cpu')] == 32
    assert L1.store_cnt[ft.MemType('cpu')] == 32
    assert L1.access_cnt[ft.MemType('cpu')] == 64
    assert L2.load_cnt[ft.MemType('cpu')] == 32
    assert L2.store_cnt[ft.MemType('cpu')] == 32
    assert L2.access_cnt[ft.MemType('cpu')] == 64
    assert S1.load_cnt[ft.MemType('cpu')] == 64
    assert S1.store_cnt[ft.MemType('cpu')] == 64
    assert S1.access_cnt[ft.MemType('cpu')] == 128


def test_access_area():
    with ft.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y", (32,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 32, label='L1') as i:
            ft.MarkLabel('S1')
            y[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)

    features = ft.structural_feature(ast)

    S1 = features[ft.lookup_id(ast, 'S1')]
    L1 = features[ft.lookup_id(ast, 'L1')]

    assert S1.load_area[ft.MemType('cpu')] == 1
    assert S1.store_area[ft.MemType('cpu')] == 1
    assert S1.access_area[ft.MemType('cpu')] == 2
    assert L1.load_area[ft.MemType('cpu')] == 32
    assert L1.store_area[ft.MemType('cpu')] == 32
    assert L1.access_area[ft.MemType('cpu')] == 64


def test_access_area_overlap():
    with ft.VarDef([
        ("x", (48,), "int32", "input", "cpu"),
        ("y1", (48,), "int32", "output", "cpu"),
        ("y2", (48,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ft.NamedScope('S1'):
            with ft.For("i", 0, 32, label='L1') as i:
                y1[i] = x[i] + 1
            with ft.For("i", 16, 48, label='L2') as i:
                y2[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)

    features = ft.structural_feature(ast)

    S1 = features[ft.lookup_id(ast, 'S1')]
    L1 = features[ft.lookup_id(ast, 'L1')]
    L2 = features[ft.lookup_id(ast, 'L2')]

    assert L1.load_area[ft.MemType('cpu')] == 32
    assert L1.store_area[ft.MemType('cpu')] == 32
    assert L1.access_area[ft.MemType('cpu')] == 64
    assert L2.load_area[ft.MemType('cpu')] == 32
    assert L2.store_area[ft.MemType('cpu')] == 32
    assert L2.access_area[ft.MemType('cpu')] == 64
    assert S1.load_area[ft.MemType('cpu')] == 48
    assert S1.store_area[ft.MemType('cpu')] == 64
    assert S1.access_area[ft.MemType('cpu')] == 112
