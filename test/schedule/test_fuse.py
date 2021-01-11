import ir
import pytest

def test_basic():
	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j1", 0, 8, nid="L2a") as j:
				y[i, j] = i + j
			with ir.For("j2", 0, 8, nid="L2b") as j:
				z[i, j] = i * j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fuse("L2a", "L2b")
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
				z[i, j] = i * j
	std = ir.pop_ast()

	assert std.match(ast)

def test_not_aligned():
	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j1", 0, 8, nid="L2a") as j:
				y[i, j] = i + j
			with ir.For("j2", 2, 10, nid="L2b") as j:
				z[i, j - 2] = i * (j - 2)
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fuse("L2a", "L2b")
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
				z[i, j] = i * j
	std = ir.pop_ast()

	assert std.match(ast)

def test_no_following():
	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("w", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z, w):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2a") as j:
				y[i, j] = i + j
			with ir.For("j", 0, 8, nid="L2b") as j:
				z[i, j] = i * j
			with ir.For("j", 0, 8, nid="L2c") as j:
				w[i, j] = i - j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.fuse("L2a", "L2c")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_different_length():
	with ir.VarDef([
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output),
			("z", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (y, z):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 8, nid="L2a") as j:
				y[i, j] = i + j
			with ir.For("j", 0, 10, nid="L2b") as j:
				z[i, j] = i * j
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.fuse("L2a", "L2b")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_dependency_unable_resolve():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.VarDef("b", (4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
				with ir.For("j", 0, 8, nid="L2a") as j:
					b[i, j] = x[i, j] * 2
				with ir.For("j", 0, 10, nid="L2b") as j:
					y[i, j] = b[i, 8 - j]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	with pytest.raises(ir.InvalidSchedule):
		s.fuse("L2a", "L2b")
	ast_ = s.ast() # Should not changed
	assert ast_.match(ast)

def test_buffer_fuse():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4, 8), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.VarDef("b", (4, 8), ir.DataType.Int32, ir.AccessType.Cache) as b:
				with ir.For("j", 0, 8, nid="L2a") as j:
					b[i, j] = x[i, j] * 2
				with ir.For("j", 0, 8, nid="L2b") as j:
					y[i, j] = b[i, j]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.fuse("L2a", "L2b")
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

