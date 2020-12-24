import ir

def test_const_fold():
	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 0 * i
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 0
	std = ir.pop_ast()

	assert std.match(ast)

def test_redundant_if():
	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 10):
				y[i] = 1
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 1
	std = ir.pop_ast()

	assert std.match(ast)

def test_different_scope():
	with ir.VarDef([
			("x", (4, 10), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				with ir.For("j", 0, 5) as j:
					with ir.If(j < 5):
						y[i] = x[i, j]
					with ir.Else():
						y[i] = x[i, j] + 2
			with ir.Else():
				with ir.For("j", 0, 10) as j:
					with ir.If(j < 5):
						y[i] = x[i, j] + 2
					with ir.Else():
						y[i] = x[i, j] + 3
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (4, 10), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				with ir.For("j", 0, 5) as j:
					y[i] = x[i, j]
			with ir.Else():
				with ir.For("j", 0, 10) as j:
					with ir.If(j < 5):
						y[i] = x[i, j] + 2
					with ir.Else():
						y[i] = x[i, j] + 3
	std = ir.pop_ast()

	assert std.match(ast)

