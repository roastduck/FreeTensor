import ir

def test_const_fold():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 0 * i
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 0
	std = ir.pop_ast()

	assert std.match(ast)

def test_partial_fold():
	# This is the case that we need a symbolic bound, instead
	# of using integers only
	with ir.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 4) as j:
				y[i, j] = 2 * j + i - j - j
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4, 4), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			with ir.For("j", 0, 4) as j:
				y[i, j] = i
	std = ir.pop_ast()

	assert std.match(ast)

def test_redundant_if():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 10):
				y[i] = 1
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 1
	std = ir.pop_ast()

	assert std.match(ast)

def test_redundant_if_2():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < i + 2):
				y[i] = 1
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 1
	std = ir.pop_ast()

	assert std.match(ast)

def test_redundant_min():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 10):
				y[i] = ir.min(i, i + 2)
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = i
	std = ir.pop_ast()

	assert std.match(ast)

def test_redundant_max():
	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 10):
				y[i] = ir.max(i, i + 2)
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("y", (4,), "int32", "output", "cpu") as y:
		with ir.For("i", 0, 4) as i:
			y[i] = i + 2
	std = ir.pop_ast()

	assert std.match(ast)

def test_different_scope():
	with ir.VarDef([
			("x", (4, 10), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (x, y):
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
			("x", (4, 10), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (x, y):
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

def test_dynamic():
	with ir.VarDef([
			("n", (), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (n, y):
		with ir.For("i", 0, n[()]) as i:
			with ir.If(n[()] + 1 > n[()]):
				y[i] = 1
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("n", (), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (n, y):
		with ir.For("i", 0, n[()]) as i:
			y[i] = 1
	std = ir.pop_ast()

	assert std.match(ast)

def test_floor_div():
	with ir.VarDef([
			("n", (), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (n, y):
		with ir.For("i", 0, n[()] // 4) as i:
			with ir.If(i * 4 < n[()]):
				y[i] = 1
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("n", (), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (n, y):
		with ir.For("i", 0, n[()] // 4) as i:
			y[i] = 1
	std = ir.pop_ast()

	assert std.match(ast)

