import ir
import pytest


def test_basic():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_func():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    func = ir.lower(ir.Func("main", ["x"], [], ir.pop_ast()), ir.CPU())
    txt = ir.dump_ast(func)
    print(txt)
    func2 = ir.load_ast(txt)
    print(func2)
    assert func2.body.match(func.body)


def test_scalar_op():
    with ir.VarDef([("x", (), "int32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = x[()] * 2 + 1
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_cast():
    with ir.VarDef([("x", (), "float32", "input", "cpu"),
                    ("y", (), "int32", "output", "cpu")]) as (x, y):
        y[()] = ir.cast(x[()], "int32") * 2
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_for():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 0, 4) as i:
            y[i] = x[i] + 1
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_reversed_for():
    with ir.VarDef([("x", (4,), "int32", "input", "cpu"),
                    ("y", (4,), "int32", "output", "cpu")]) as (x, y):
        with ir.For("i", 3, -1, -1) as i:
            y[i] = x[i] + 1
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_if():
    with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ir.For("i", 0, 4) as i:
            with ir.If(i < 2):
                y[i] = 0
            with ir.Else():
                y[i] = 1
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_id():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        ir.MarkNid("foo")
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ir.Schedule(ast2)
    assert s.find("foo").type() == ir.ASTNodeType.Store


def test_id_of_stmt_seq():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        with ir.NamedScope("foo"):
            x[2, 3] = 2.0
            x[1, 0] = 3.0
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ir.Schedule(ast2)
    assert s.find("foo").type() == ir.ASTNodeType.StmtSeq


def test_complex_name():
    with ir.VarDef("x!@#$%^&*", (4, 4), "float32", "output", "cpu") as x:
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)


def test_complex_id():
    with ir.VarDef("x", (4, 4), "float32", "output", "cpu") as x:
        ir.MarkNid("id!@#$%^&*")
        x[2, 3] = 2.0
        x[1, 0] = 3.0
    ast = ir.pop_ast()
    txt = ir.dump_ast(ast)
    print(txt)
    ast2 = ir.load_ast(txt)
    print(ast2)
    assert ast2.match(ast)
    s = ir.Schedule(ast2)
    assert s.find("id!@#$%^&*").type() == ir.ASTNodeType.Store
