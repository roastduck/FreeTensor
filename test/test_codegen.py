import ir
import numpy as np

def test_hello_world():
	with ir.VarDef("x", (4, 4), ir.DataType.Float32, ir.AccessType.Output) as x:
		x[2, 3] = 2.0
		x[1, 0] = 3.0

	ast = ir.lower(ir.pop_ast())
	print(ast)
	code, params = ir.code_gen_c(ast)
	print(code)

	x = np.zeros((4, 4), dtype="float32")
	driver = ir.Driver(code, params)
	driver.set_params({"x": x})
	driver.run()

	x_std = np.zeros((4, 4), dtype="float32")
	x_std[2, 3] = 2.0
	x_std[1, 0] = 3.0
	assert np.array_equal(x, x_std)

def test_scalar_op():
	with ir.VarDef([
			("x", (), ir.DataType.Int32, ir.AccessType.Input),
			("y", (), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		y[()] = x[()] * 2 + 1

	code, params = ir.code_gen_c(ir.lower(ir.pop_ast()))
	print(code)
	x = np.array(5, dtype="int32")
	y = np.array(0, dtype="int32")
	driver = ir.Driver(code, params)
	driver.set_params({"x": x, "y": y})
	driver.run()

	assert y[()] == 11

def test_for():
	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			y[i] = x[i] + 1

	code, params = ir.code_gen_c(ir.lower(ir.pop_ast()))
	print(code)
	x = np.array([1, 2, 3, 4], dtype="int32")
	y = np.zeros((4,), dtype="int32")
	driver = ir.Driver(code, params)
	driver.set_params({"x": x, "y": y})
	driver.run()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y, y_std)

def test_if():
	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				y[i] = 0
			with ir.Else():
				y[i] = 1

	code, params = ir.code_gen_c(ir.lower(ir.pop_ast()))
	print(code)
	y = np.zeros((4,), dtype="int32")
	driver = ir.Driver(code, params)
	driver.set_params({"y": y})
	driver.run()

	y_std = np.array([0, 0, 1, 1], dtype="int32")
	assert np.array_equal(y, y_std)

def test_tiling():
	with ir.VarDef([
			("a", (256, 256), ir.DataType.Float32, ir.AccessType.Input),
			("b", (256, 256), ir.DataType.Float32, ir.AccessType.Input),
			("c", (256, 256), ir.DataType.Float32, ir.AccessType.Output)]) as (a, b, c):
		with ir.For("i", 0, 256, nid="Li") as i:
			with ir.For("j", 0, 256, nid="Lj") as j:
				with ir.NamedScope("S0"):
					c[i, j] = 0
					with ir.For("k", 0, 256, nid="Lk") as k:
						ir.MarkNid("S1")
						c[i, j] = c[i, j] + a[i, k] * b[k, j]

	i, j, S1 = "Li", "Lj", "S1"

	s = ir.Schedule(ir.pop_ast())
	i0, i1 = s.split(i, 32)
	j0, j1 = s.split(j, 32)
	s.reorder([i0, j0, i1, j1])

	s.cache_write("S0", "c")

	load_a, _ = s.cache_read(S1, "a")
	s.move_to(load_a, j0, to_begin=True)

	load_b, _ = s.cache_read(S1, "b")
	s.move_to(load_b, j0, to_begin=True)

	ast = ir.lower(s.ast())
	print(ast)

	with ir.VarDef([
			("a", (256, 256), ir.DataType.Float32, ir.AccessType.Input),
			("b", (256, 256), ir.DataType.Float32, ir.AccessType.Input),
			("c", (256, 256), ir.DataType.Float32, ir.AccessType.Output)]) as (a, b, c):
		with ir.For("i.0", 0, 8) as i0:
			with ir.For("j.0", 0, 8) as j0:
				with ir.VarDef([
					("a.r", (32, 256, 1, 1), ir.DataType.Float32, ir.AccessType.Cache),
					("b.r", (32, 256, 1, 1), ir.DataType.Float32, ir.AccessType.Cache)]) as (ar, br):
					with ir.For("i.1", 0, 32) as i1:
						with ir.For("k", 0, 256) as k:
							ir.Any()
					with ir.For("j.1", 0, 32) as i1:
						with ir.For("k", 0, 256) as k:
							ir.Any()
					with ir.For("i.1", 0, 32) as i1:
						with ir.For("j.1", 0, 32) as i1:
							with ir.VarDef("c.w", (1, 1), ir.DataType.Float32, ir.AccessType.Cache) as cw:
								cw[0, 0] = 0
								with ir.For("k", 0, 256) as k:
									ir.Any()
								ir.Any()
	assert ir.pop_ast().match(ast)

	code, params = ir.code_gen_c(ast)
	print(code)
	a = np.random.rand(256, 256).astype("float32")
	b = np.random.rand(256, 256).astype("float32")
	c = np.zeros((256, 256), dtype="float32")
	driver = ir.Driver(code, params)
	driver.set_params({"a": a, "b": b, "c": c})
	driver.run()

	c_std = a @ b
	assert np.all(np.isclose(c, c_std))

