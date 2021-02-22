import ir

def test_basic():
	with ir.VarDef("y", (), "int32", "output", "cpu") as y:
		y[()] = 1
		y[()] = 2
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (), "int32", "output", "cpu") as y:
		y[()] = 2
	std = ir.pop_ast()

	assert std.match(ast)

def test_one_then_many():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		y[0] = 1
		with ir.For("i", 0, 4) as i:
			y[i] = i
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = i
	std = ir.pop_ast()

	assert std.match(ast)

def test_many_then_one_no_remove():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = i
		y[0] = 1
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = i
		y[0] = 1
	std = ir.pop_ast()

	assert std.match(ast)

def test_write_then_reduce():
	with ir.VarDef("y", (), "int32", "output", "cpu") as y:
		y[()] = 1
		y[()] = y[()] + 2
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (), "int32", "output", "cpu") as y:
		y[()] = 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_reduce_then_reduce():
	with ir.VarDef("y", (), "int32", "inout", "cpu") as y:
		y[()] = y[()] + 1
		y[()] = y[()] + 2
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (), "int32", "inout", "cpu") as y:
		y[()] = y[()] + 3
	std = ir.make_reduction(ir.pop_ast())

	assert std.match(ast)

