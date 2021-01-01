import ir
import pytest

def test_basic():
	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i, j] = i + j
				z[i, j] = i * j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fission("L2", "S0")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 8) as j:
				y[i, j] = i + j
			with ir.For("j", 0, 8) as j:
				z[i, j] = i * j
	std = ir.pop_ast()

	assert std.match(ast)

def test_buffer_hoist():
	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				with ir.VarDef("buf", (4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
					ir.MarkNid("S0")
					b[i, j] = x0[i, j] + x1[i, j]
					y[i, j] = b[i, j] * b[i, j]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fission("L2", "S0")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y):
		with ir.For("i", 0, 4) as i:
			with ir.VarDef("buf", (4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
				with ir.For("j", 0, 8) as j:
					b[i, j] = x0[i, j] + x1[i, j]
				with ir.For("j", 0, 8) as j:
					y[i, j] = b[i, j] * b[i, j]
	std = ir.pop_ast()

	assert std.match(ast)

def test_buffer_no_hoist():
	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y, z):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				with ir.VarDef("buf", (4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
					b[i, j] = x0[i, j] + x1[i, j]
					ir.MarkNid("S0")
					y[i, j] = b[i, j] * b[i, j]
					z[i, j] = x0[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fission("L2", "S0")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y, z):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 8) as j:
				with ir.VarDef("buf", (4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
					b[i, j] = x0[i, j] + x1[i, j]
					y[i, j] = b[i, j] * b[i, j]
			with ir.For("j", 0, 8) as j:
				z[i, j] = x0[i, j] * 2
	std = ir.pop_ast()

	assert std.match(ast)

def test_correct_dependency_basic():
	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				with ir.VarDef("buf", (1,), ir.DataType.Int32, ir.AccessType.Cache) as b:
					ir.MarkNid("S0")
					b[0] = x0[i, j] + x1[i, j]
					y[i, j] = b[0] * b[0]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fission("L2", "S0")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y):
		with ir.For("i", 0, 4) as i:
			with ir.VarDef("buf", (1, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
				with ir.For("j", 0, 8) as j:
					b[0, j] = x0[i, j] + x1[i, j]
				with ir.For("j", 0, 8) as j:
					y[i, j] = b[0, j] * b[0, j]
	std = ir.pop_ast()

	assert std.match(ast)

def test_correct_dependency_multi_loop():
	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				with ir.VarDef("buf", (1,), ir.DataType.Int32, ir.AccessType.Cache) as b:
					ir.MarkNid("S0")
					b[0] = x0[i, j] + x1[i, j]
					y[i, j] = b[0] * b[0]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fission("L1", "S0")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x0, x1, y):
		with ir.VarDef("buf", (1, 4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
			with ir.For("i", 0, 4) as i:
				with ir.For("j", 0, 8) as j:
					b[0, i, j] = x0[i, j] + x1[i, j]
			with ir.For("i", 0, 4) as i:
				with ir.For("j", 0, 8) as j:
					y[i, j] = b[0, i, j] * b[0, i, j]
	std = ir.pop_ast()

	assert std.match(ast)

def test_correct_dependency_real_dep():
	with ir.VarDef([
			("x", (4), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.VarDef("buf", (1,), ir.DataType.Int32, ir.AccessType.Cache) as b:
				ir.MarkNid("S0")
				b[0] = x[i] * 2
				with ir.For("j", 0, 8, nid="L2") as j:
					y[i, j] = b[0] * b[0]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fission("L1", "S0")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.VarDef("buf", (1, 4), ir.DataType.Int32, ir.AccessType.Cache) as b:
			with ir.For("i", 0, 4) as i:
				b[0, i] = x[i] * 2
			with ir.For("i", 0, 4) as i:
				with ir.For("j", 0, 8) as j:
					y[i, j] = b[0, i] * b[0, i]
	std = ir.pop_ast()

	assert std.match(ast)

def test_correct_dependency_unable_resolve():
	with ir.VarDef([
			("x0", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("x1", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("buf", (1,), ir.DataType.Int32, ir.AccessType.InOut)]) as (x0, x1, y, b):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				b[0] = x0[i, j] + x1[i, j]
				y[i, j] = b[0] * b[0]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.fission("L2", "S0")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

