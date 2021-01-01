import ir

def test_sink_stmt_seq():
	with ir.VarDef([
			("x", (5,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.VarDef("b", (1,), ir.DataType.Int32, ir.AccessType.Cache) as b:
			with ir.For("i", 0, 4) as i:
				b[0] = x[i] + x[i + 1]
				y[i] = b[0] * i
			y[0] = 0
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("x", (5,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.VarDef("b", (1,), ir.DataType.Int32, ir.AccessType.Cache) as b:
			with ir.For("i", 0, 4) as i:
				b[0] = x[i] + x[i + 1]
				y[i] = b[0] * i
		y[0] = 0
	std = ir.pop_ast()

	assert std.match(ast)

