import ir
import numpy as np

def test_hello_world():
	with ir.VarDef("x", (4, 4), ir.DataType.Float32, ir.AccessType.Output) as x:
		x[2, 3] = 2.0
		x[1, 0] = 3.0

	ast = ir.lower(ir.pop_ast())
	print(ast)
	code, params = ir.codegen(ast, ir.CPU())
	print(code)

	x_np = np.zeros((4, 4), dtype="float32")
	x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params)
	driver.set_params({"x": x_arr})
	driver.run()
	x_np = x_arr.numpy()

	x_std = np.zeros((4, 4), dtype="float32")
	x_std[2, 3] = 2.0
	x_std[1, 0] = 3.0
	assert np.array_equal(x_np, x_std)

def test_scalar_op():
	with ir.VarDef([
			("x", (), ir.DataType.Int32, ir.AccessType.Input),
			("y", (), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		y[()] = x[()] * 2 + 1

	code, params = ir.codegen(ir.lower(ir.pop_ast()), ir.CPU())
	print(code)
	x_np = np.array(5, dtype="int32")
	y_np = np.array(0, dtype="int32")
	x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
	y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	assert y_np[()] == 11

def test_for():
	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			y[i] = x[i] + 1

	code, params = ir.codegen(ir.lower(ir.pop_ast()), ir.CPU())
	print(code)
	x_np = np.array([1, 2, 3, 4], dtype="int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
	y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_if():
	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 2):
				y[i] = 0
			with ir.Else():
				y[i] = 1

	code, params = ir.codegen(ir.lower(ir.pop_ast()), ir.CPU())
	print(code)
	y_np = np.zeros((4,), dtype="int32")
	y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params)
	driver.set_params({"y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([0, 0, 1, 1], dtype="int32")
	assert np.array_equal(y_np, y_std)

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

	code, params = ir.codegen(ast, ir.CPU())
	print(code)
	a_np = np.random.rand(256, 256).astype("float32")
	b_np = np.random.rand(256, 256).astype("float32")
	c_np = np.zeros((256, 256), dtype="float32")
	a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
	b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
	c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params)
	driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
	driver.run()
	c_np = c_arr.numpy()

	c_std = a_np @ b_np
	assert np.all(np.isclose(c_np, c_std))

