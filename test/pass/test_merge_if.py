import ir

def test_basic():
	with ir.VarDef([
			("y1", (4,), "int32", "output", "cpu"),
			("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				y1[i] = 0
			with ir.Else():
				y1[i] = 1
			with ir.If(i < 2):
				y2[i] = 2
			with ir.Else():
				y2[i] = 3
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (4,), "int32", "output", "cpu"),
			("y2", (4,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				y1[i] = 0
				y2[i] = 2
			with ir.Else():
				y1[i] = 1
				y2[i] = 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_no_merge_different_cond():
	with ir.VarDef([
			("y1", (5,), "int32", "output", "cpu"),
			("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 5) as i:
			with ir.If(i < 2):
				y1[i] = 0
			with ir.Else():
				y1[i] = 1
			with ir.If(i < 3):
				y2[i] = 2
			with ir.Else():
				y2[i] = 3
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef([
			("y1", (5,), "int32", "output", "cpu"),
			("y2", (5,), "int32", "output", "cpu")]) as (y1, y2):
		with ir.For("i", 0, 5) as i:
			with ir.If(i < 2):
				y1[i] = 0
			with ir.Else():
				y1[i] = 1
			with ir.If(i < 3):
				y2[i] = 2
			with ir.Else():
				y2[i] = 3
	std = ir.pop_ast()

	assert std.match(ast)

def test_no_merge_may_update():
	with ir.VarDef("a", (4,), "int32", "inout", "cpu") as a:
		with ir.For("i", 0, 4) as i:
			with ir.If(a[i] > 10):
				a[i] = a[i] / 2
			with ir.If(a[i] > 10):
				a[i] = a[i] / 2
	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	with ir.VarDef("a", (4,), "int32", "inout", "cpu") as a:
		with ir.For("i", 0, 4) as i:
			with ir.If(a[i] > 10):
				a[i] = a[i] / 2
			with ir.If(a[i] > 10):
				a[i] = a[i] / 2
	std = ir.pop_ast()

	assert std.match(ast)

