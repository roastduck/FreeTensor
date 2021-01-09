import ir
import pytest

def test_pure_swap():
	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y3", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y4", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2, y3, y4):
		with ir.For("i", 0, 4, nid="L1") as i:
			ir.MarkNid("S1")
			y1[i] = i + 1
			y2[i] = i + 2
			y3[i] = i + 3
			ir.MarkNid("S2")
			y4[i] = i + 4
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.move_to("S2", "S1")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y3", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y4", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2, y3, y4):
		with ir.For("i", 0, 4) as i:
			y1[i] = i + 1
			y4[i] = i + 4
			y2[i] = i + 2
			y3[i] = i + 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_swap_to_begin():
	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y3", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y4", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2, y3, y4):
		with ir.For("i", 0, 4, nid="L1") as i:
			y1[i] = i + 1
			y2[i] = i + 2
			y3[i] = i + 3
			ir.MarkNid("S1")
			y4[i] = i + 4
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.move_to("S1", "L1", to_begin=True)
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y3", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y4", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2, y3, y4):
		with ir.For("i", 0, 4) as i:
			y4[i] = i + 4
			y1[i] = i + 1
			y2[i] = i + 2
			y3[i] = i + 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_pure_fission():
	with ir.VarDef([
			("y1", (4, 4), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4, 4), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 4, nid="L2") as j:
				ir.MarkNid("S1")
				y1[i, j] = i * j + 1
				y2[i, j] = i * j + 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.move_to("S1", "L1", to_begin=True)
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4, 4), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4, 4), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 4) as j:
				y1[i, j] = i * j + 1
			with ir.For("j", 0, 4) as j:
				y2[i, j] = i * j + 2
	std = ir.pop_ast()

	assert std.match(ast)

def test_swap_and_fission():
	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4, 4), ir.DataType.Int32, ir.AccessType.Output),
			("y3", (4, 4), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2, y3):
		with ir.For("i", 0, 4, nid="L1") as i:
			ir.MarkNid("S1")
			y1[i] = i
			with ir.For("j", 0, 4, nid="L2") as j:
				ir.MarkNid("S2")
				y2[i, j] = i * j + 1
				ir.MarkNid("S3")
				y3[i, j] = i * j + 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.move_to("S3", "L1", to_begin=True)
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4, 4), ir.DataType.Int32, ir.AccessType.Output),
			("y3", (4, 4), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2, y3):
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 4) as j:
				y3[i, j] = i * j + 2
			y1[i] = i
			with ir.For("j", 0, 4) as j:
				y2[i, j] = i * j + 1
	std = ir.pop_ast()

	assert std.match(ast)

def test_crossing_var_def():
	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4, 4), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2):
		with ir.For("i", 0, 4, nid="L1") as i:
			y1[i] = i
			with ir.For("j", 0, 4, nid="L2") as j:
				with ir.VarDef("t", (), ir.DataType.Int32, ir.AccessType.Cache) as t:
					ir.MarkNid("S1")
					t[()] = i * j
					y2[i, j] = t[()]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.move_to("S1", "L1", to_begin=True)
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4,), ir.DataType.Int32, ir.AccessType.Output),
			("y2", (4, 4), ir.DataType.Int32, ir.AccessType.Output)]) as (y1, y2):
		with ir.For("i", 0, 4) as i:
			with ir.VarDef("t", (4,), ir.DataType.Int32, ir.AccessType.Cache) as t:
				with ir.For("j", 0, 4) as j:
					t[j] = i * j
				y1[i] = i
				with ir.For("j", 0, 4) as j:
					y2[i, j] = t[j]
	std = ir.pop_ast()

	assert std.match(ast)

