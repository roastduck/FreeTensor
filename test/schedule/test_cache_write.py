import ir
import pytest

def test_basic():
	with ir.VarDef([
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_write("S0", "y", "cpu")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 8) as j:
				with ir.VarDef("b", (1, 1), "int32", "cache", "cpu") as b:
					b[0, 0] = x[i, j] * 2
					y[i, j] = b[0, 0]
	std = ir.pop_ast()

	assert std.match(ast)

def test_reduction():
	with ir.VarDef([
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i, j] = y[i, j] + x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_write("S0", "y", "cpu")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "inout", "cpu")]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 8) as j:
				with ir.VarDef("b", (1, 1), "int32", "cache", "cpu") as b:
					b[0, 0] = 2 * x[i, j] # After remove_writes pass
					y[i, j] = y[i, j] + b[0, 0]
	std = ir.make_reduction(ir.pop_ast())

	assert std.match(ast)

def test_different_indices():
	with ir.VarDef([
			("x", (4,), "int32", "input", "cpu"),
			("y", (5,), "int32", "output", "cpu")]) as (x, y):
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
	s.cache_write("S0", "y", "cpu")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4,), "int32", "input", "cpu"),
			("y", (5,), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.VarDef("b", (2,), "int32", "cache", "cpu") as b:
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
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.cache_write("S0", "x", "cpu")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_no_stmt():
	with ir.VarDef([
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.cache_write("S0", "y", "cpu")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_local_var_as_index():
	with ir.VarDef([
			("x", (4, 8), "int32", "input", "cpu"),
			("y", (4, 8), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				y[i, j] = x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.cache_write("L2", "x", "cpu")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

