import ir


def test_flop():
    with ir.VarDef([
        ("x1", (32,), "float32", "input", "cpu"),
        ("x2", (32,), "float32", "input", "cpu"),
        ("x3", (32,), "float32", "input", "cpu"),
        ("y", (32,), "float32", "output", "cpu"),
    ]) as (x1, x2, x3, y):
        with ir.For("i", 0, 32, nid='L1') as i:
            ir.MarkNid('S1')
            y[i] = x1[i] * x2[i] + x3[i]
    ast = ir.pop_ast()
    print(ast)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ir.structural_feature(ast).items()))

    assert features['S1'].op_cnt[ir.parseDType('float32')] == 2
    assert features['L1'].op_cnt[ir.parseDType('float32')] == 64


def test_access_count():
    with ir.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y", (32,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 32, nid='L1') as i:
            ir.MarkNid('S1')
            y[i] = x[i] + 1
    ast = ir.pop_ast()
    print(ast)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ir.structural_feature(ast).items()))

    assert features['S1'].load_cnt[ir.parseMType('cpu')] == 1
    assert features['S1'].store_cnt[ir.parseMType('cpu')] == 1
    assert features['S1'].access_cnt[ir.parseMType('cpu')] == 2
    assert features['L1'].load_cnt[ir.parseMType('cpu')] == 32
    assert features['L1'].store_cnt[ir.parseMType('cpu')] == 32
    assert features['L1'].access_cnt[ir.parseMType('cpu')] == 64


def test_access_count_overlap():
    with ir.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y1", (32,), "int32", "output", "cpu"),
        ("y2", (32,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.NamedScope('S1'):
            with ir.For("i", 0, 32, nid='L1') as i:
                y1[i] = x[i] + 1
            with ir.For("i", 16, 48, nid='L2') as i:
                y2[i] = x[i] + 1
    ast = ir.pop_ast()
    print(ast)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ir.structural_feature(ast).items()))

    assert features['L1'].load_cnt[ir.parseMType('cpu')] == 32
    assert features['L1'].store_cnt[ir.parseMType('cpu')] == 32
    assert features['L1'].access_cnt[ir.parseMType('cpu')] == 64
    assert features['L2'].load_cnt[ir.parseMType('cpu')] == 32
    assert features['L2'].store_cnt[ir.parseMType('cpu')] == 32
    assert features['L2'].access_cnt[ir.parseMType('cpu')] == 64
    assert features['S1'].load_cnt[ir.parseMType('cpu')] == 64
    assert features['S1'].store_cnt[ir.parseMType('cpu')] == 64
    assert features['S1'].access_cnt[ir.parseMType('cpu')] == 128


def test_access_area():
    with ir.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y", (32,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 32, nid='L1') as i:
            ir.MarkNid('S1')
            y[i] = x[i] + 1
    ast = ir.pop_ast()
    print(ast)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ir.structural_feature(ast).items()))

    assert features['S1'].load_area[ir.parseMType('cpu')] == 1
    assert features['S1'].store_area[ir.parseMType('cpu')] == 1
    assert features['S1'].access_area[ir.parseMType('cpu')] == 2
    assert features['L1'].load_area[ir.parseMType('cpu')] == 32
    assert features['L1'].store_area[ir.parseMType('cpu')] == 32
    assert features['L1'].access_area[ir.parseMType('cpu')] == 64


def test_access_area_overlap():
    with ir.VarDef([
        ("x", (32,), "int32", "input", "cpu"),
        ("y1", (32,), "int32", "output", "cpu"),
        ("y2", (32,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.NamedScope('S1'):
            with ir.For("i", 0, 32, nid='L1') as i:
                y1[i] = x[i] + 1
            with ir.For("i", 16, 48, nid='L2') as i:
                y2[i] = x[i] + 1
    ast = ir.pop_ast()
    print(ast)

    features = dict(
        map(lambda kv: (str(kv[0]), kv[1]),
            ir.structural_feature(ast).items()))

    assert features['L1'].load_area[ir.parseMType('cpu')] == 32
    assert features['L1'].store_area[ir.parseMType('cpu')] == 32
    assert features['L1'].access_area[ir.parseMType('cpu')] == 64
    assert features['L2'].load_area[ir.parseMType('cpu')] == 32
    assert features['L2'].store_area[ir.parseMType('cpu')] == 32
    assert features['L2'].access_area[ir.parseMType('cpu')] == 64
    assert features['S1'].load_area[ir.parseMType('cpu')] == 48
    assert features['S1'].store_area[ir.parseMType('cpu')] == 64
    assert features['S1'].access_area[ir.parseMType('cpu')] == 112
