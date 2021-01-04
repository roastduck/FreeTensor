import ir

def test_basic():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = 0
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				y[i] = y[i] + x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_read("S0", "x")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			y[i] = 0
			with ir.For("j", 0, 8) as j:
				with ir.VarDef("b", (1, 1), ir.DataType.Int32, ir.AccessType.Cache) as b:
					b[0, 0] = x[i, j]
					y[i] = y[i] + b[0, 0] * 2
	std = ir.pop_ast()

	assert std.match(ast)

def test_different_indices():
	with ir.VarDef([
			("x", (5,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			ir.MarkNid("S0")
			y[i] = x[i] + x[i + 1]
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_read("S0", "x")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (5,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.VarDef("b", (2,), ir.DataType.Int32, ir.AccessType.Cache) as b:
				ir.Any()
				ir.Any() # b[0] and b[1] in any order
				y[i] = b[0] + b[1]
	std = ir.pop_ast()

	assert std.match(ast)

def test_block():
	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = 0
			with ir.For("j", 0, 8, nid="L2") as j:
				ir.MarkNid("S0")
				with ir.If(x[i, j] < 0):
					y[i] = y[i] + x[i, j]
				with ir.Else():
					y[i] = y[i] + x[i, j] * 2
	ast = ir.pop_ast()
	print(ast)
	s = ir.Schedule(ast)
	s.cache_read("S0", "x")
	ast = s.ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4, 8), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			y[i] = 0
			with ir.For("j", 0, 8) as j:
				with ir.VarDef("b", (1, 1), ir.DataType.Int32, ir.AccessType.Cache) as b:
					b[0, 0] = x[i, j]
					with ir.If(b[0, 0] < 0):
						y[i] = y[i] + b[0, 0]
					with ir.Else():
						y[i] = y[i] + b[0, 0] * 2
	std = ir.pop_ast()

	assert std.match(ast)

