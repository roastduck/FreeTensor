import ir
import pytest

def test_basic():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_write("S0", "y")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 8) as j:
				with ir.VarDef("b", (1, 1), ir.DataType.Int32, ir.AccessType.Cache) as b:
					b[0, 0] = x[i, j] * 2
					y[i, j] = b[0, 0]
	std = ir.pop_ast()

	assert std.match(ast)

def test_different_indices():
	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (5,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			ir.MarkNid("S0")
			with ir.If(x[i] < 0):
				y[i] = x[i]
				y[i + 1] = x[i] * 2
			with ir.Else():
				y[i] = x[i] * 2
				y[i + 1] = x[i]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_write("S0", "y")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (5,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.VarDef("b", (2,), ir.DataType.Int32, ir.AccessType.Cache) as b:
				with ir.If(x[i] < 0):
					b[0] = x[i]
					b[1] = x[i] * 2
				with ir.Else():
					b[0] = x[i] * 2
					b[1] = x[i]
				y[i] = b[0]
				y[i + 1] = b[1]
	std = ir.pop_ast()

	assert std.match(ast)

def test_no_store():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.cache_write("S0", "x")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_no_stmt():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.cache_write("S0", "y")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

