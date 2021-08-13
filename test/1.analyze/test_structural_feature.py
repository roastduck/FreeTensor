import ir


def test_access_area():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y", (4,), "int32", "output", "cpu"),
    ]) as (x, y):
        with ir.For("i", 0, 32, nid='L1') as i:
            ir.MarkNid('S1')
            y[i] = x[i] + 1
    ast = ir.pop_ast()
    print(ast)

    features = ir.structural_feature(ast)

    assert features['S1'].loadArea[ir.parseMType('cpu')] == 1
    assert features['S1'].storeArea[ir.parseMType('cpu')] == 1
    assert features['S1'].accessArea[ir.parseMType('cpu')] == 2
    assert features['L1'].loadArea[ir.parseMType('cpu')] == 32
    assert features['L1'].storeArea[ir.parseMType('cpu')] == 32
    assert features['L1'].accessArea[ir.parseMType('cpu')] == 64


def test_access_area_overlap():
    with ir.VarDef([
        ("x", (4,), "int32", "input", "cpu"),
        ("y1", (4,), "int32", "output", "cpu"),
        ("y2", (4,), "int32", "output", "cpu"),
    ]) as (x, y1, y2):
        with ir.NamedScope('S1'):
            with ir.For("i", 0, 32, nid='L1') as i:
                y1[i] = x[i] + 1
            with ir.For("i", 16, 48, nid='L2') as i:
                y2[i] = x[i] + 1
    ast = ir.pop_ast()
    print(ast)

    features = ir.structural_feature(ast)

    assert features['L1'].loadArea[ir.parseMType('cpu')] == 32
    assert features['L1'].storeArea[ir.parseMType('cpu')] == 32
    assert features['L1'].accessArea[ir.parseMType('cpu')] == 64
    assert features['L2'].loadArea[ir.parseMType('cpu')] == 32
    assert features['L2'].storeArea[ir.parseMType('cpu')] == 32
    assert features['L2'].accessArea[ir.parseMType('cpu')] == 64
    assert features['S1'].loadArea[ir.parseMType('cpu')] == 48
    assert features['S1'].storeArea[ir.parseMType('cpu')] == 64
    assert features['S1'].accessArea[ir.parseMType('cpu')] == 112
