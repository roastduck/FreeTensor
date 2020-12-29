import ir
import pytest

def test_basic():
	with ir.VarDef("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				y[i, j] = i + j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.merge("L1", "L2")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 32) as i:
			y[i / 8, i % 8] = i / 8 + i % 8
	std = ir.pop_ast()
	print(std)

	assert std.match(ast)

def test_invalid():
	with ir.VarDef("y", (4, 4, 4), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 4, nid="L2") as j:
				with ir.For("k", 0, 4, nid="L3") as k:
					y[i, j, k] = i + j + k
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.merge("L1", "L3")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_if_in_between():
	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.If(x[i] > 0):
				with ir.For("j", 0, 8, nid="L2") as j:
					y[i, j] = i + j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.merge("L1", "L2")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 32) as i:
			with ir.If(x[i / 8] > 0):
				y[i / 8, i % 8] = i / 8 + i % 8
	std = ir.pop_ast()

	assert std.match(ast)

def test_stmt_in_between():
	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 4, nid="L1") as i:
			z[i] = i
			with ir.For("j", 0, 8, nid="L2") as j:
				y[i, j] = i + j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.merge("L1", "L2")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 32) as i:
			with ir.If(i % 8 == 0):
				z[i / 8] = i / 8
			y[i / 8, i % 8] = i / 8 + i % 8
	std = ir.pop_ast()

	assert std.match(ast)

