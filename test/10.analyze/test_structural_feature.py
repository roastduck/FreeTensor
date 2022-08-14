import freetensor as ft


def label_to_id(label, ast):
    return str(ft.Schedule(ast).find(label).id)


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

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ft.structural_feature(ast).items()))

    assert features[label_to_id('S1', ast)].op_cnt[ft.DataType('float32')] == 2
    assert features[label_to_id('L1', ast)].op_cnt[ft.DataType('float32')] == 64


def test_access_count():
    with ft.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y", (32,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 32, label='L1') as i:
            ft.MarkLabel('S1')
            y[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ft.structural_feature(ast).items()))

    assert features[label_to_id('S1', ast)].load_cnt[ft.MemType('cpu')] == 1
    assert features[label_to_id('S1', ast)].store_cnt[ft.MemType('cpu')] == 1
    assert features[label_to_id('S1', ast)].access_cnt[ft.MemType('cpu')] == 2
    assert features[label_to_id('L1', ast)].load_cnt[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].store_cnt[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].access_cnt[ft.MemType('cpu')] == 64


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

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ft.structural_feature(ast).items()))

    assert features[label_to_id('L1', ast)].load_cnt[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].store_cnt[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].access_cnt[ft.MemType('cpu')] == 64
    assert features[label_to_id('L2', ast)].load_cnt[ft.MemType('cpu')] == 32
    assert features[label_to_id('L2', ast)].store_cnt[ft.MemType('cpu')] == 32
    assert features[label_to_id('L2', ast)].access_cnt[ft.MemType('cpu')] == 64
    assert features[label_to_id('S1', ast)].load_cnt[ft.MemType('cpu')] == 64
    assert features[label_to_id('S1', ast)].store_cnt[ft.MemType('cpu')] == 64
    assert features[label_to_id('S1', ast)].access_cnt[ft.MemType('cpu')] == 128


def test_access_area():
    with ft.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y", (32,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ft.For("i", 0, 32, label='L1') as i:
            ft.MarkLabel('S1')
            y[i] = x[i] + 1
    ast = ft.pop_ast(verbose=True)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ft.structural_feature(ast).items()))

    assert features[label_to_id('S1', ast)].load_area[ft.MemType('cpu')] == 1
    assert features[label_to_id('S1', ast)].store_area[ft.MemType('cpu')] == 1
    assert features[label_to_id('S1', ast)].access_area[ft.MemType('cpu')] == 2
    assert features[label_to_id('L1', ast)].load_area[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].store_area[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].access_area[ft.MemType('cpu')] == 64


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

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ft.structural_feature(ast).items()))

    assert features[label_to_id('L1', ast)].load_area[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].store_area[ft.MemType('cpu')] == 32
    assert features[label_to_id('L1', ast)].access_area[ft.MemType('cpu')] == 64
    assert features[label_to_id('L2', ast)].load_area[ft.MemType('cpu')] == 32
    assert features[label_to_id('L2', ast)].store_area[ft.MemType('cpu')] == 32
    assert features[label_to_id('L2', ast)].access_area[ft.MemType('cpu')] == 64
    assert features[label_to_id('S1', ast)].load_area[ft.MemType('cpu')] == 48
    assert features[label_to_id('S1', ast)].store_area[ft.MemType('cpu')] == 64
    assert features[label_to_id('S1',
                                ast)].access_area[ft.MemType('cpu')] == 112
