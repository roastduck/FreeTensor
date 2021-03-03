import ir
import numpy as np

target = ir.CPU()
device = ir.Device(target)

def test_omp_for():
	with ir.VarDef([
			("x", (4,), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "openmp")
	ast = ir.lower(s.ast(), target)
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([1, 2, 3, 4], dtype="int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, ir.Device(target))
	y_arr = ir.Array(y_np, ir.Device(target))
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_parallel_reduction():
	with ir.VarDef([
			("x", (4, 64), "int32", "input", "cpu"),
			("y", (4,), "int32", "inout", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 64, nid="L2") as j:
				y[i] = y[i] + x[i, j]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L2", "openmp")
	ast = ir.lower(s.ast(), target)
	print(ast)

	code, params = ir.codegen(ast, target)
	assert "#pragma omp atomic" in code
	assert "+=" in code
	print(code)
	x_np = np.random.rand(4, 64).astype("int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.sum(x_np, axis=1)
	assert np.array_equal(y_np, y_std)

def test_serial_reduction():
	with ir.VarDef([
			("x", (4, 64), "int32", "input", "cpu"),
			("y", (4,), "int32", "inout", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 64, nid="L2") as j:
				y[i] = y[i] + x[i, j]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "openmp")
	ast = ir.lower(s.ast(), target)
	print(ast)

	code, params = ir.codegen(ast, target)
	assert "#pragma omp atomic" not in code
	assert "+=" in code
	print(code)
	x_np = np.random.rand(4, 64).astype("int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.sum(x_np, axis=1)
	assert np.array_equal(y_np, y_std)

def test_unroll_for():
	with ir.VarDef([
			("x", (4,), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.unroll("L1", 4)
	ast = ir.lower(s.ast(), target)
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([1, 2, 3, 4], dtype="int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, ir.Device(ir.CPU()))
	y_arr = ir.Array(y_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params, ir.Device(ir.CPU()))
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y_np, y_std)