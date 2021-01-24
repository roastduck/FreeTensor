import ir
import numpy as np

target = ir.GPU()
# TODO: Detect GPU arch and set it to target
device = ir.Device(target)

def test_basic():
	with ir.VarDef([
			("x", (4,), "int32", "input", "gpuglobal"),
			("y", (4,), "int32", "output", "gpuglobal")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast())
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([1, 2, 3, 4], dtype="int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_shmem():
	with ir.VarDef([
			("x", (4,), "int32", "input", "gpuglobal"),
			("y", (4,), "int32", "output", "gpuglobal")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			ir.MarkNid("S1")
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	load_x, _ = s.cache_read("S1", "x", "gpushared")
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast())
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	assert "__shared__" in code
	x_np = np.array([1, 2, 3, 4], dtype="int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_global_mem():
	with ir.VarDef([
			("x", (4,), "int32", "input", "gpuglobal"),
			("y", (4,), "int32", "output", "gpuglobal")]) as (x, y):
		with ir.VarDef("t", (4,), "int32", "cache", "gpuglobal") as t:
			with ir.For("i", 0, 4, nid="L1") as i:
				t[i] = x[i] * 2
			with ir.For("i", 0, 4, nid="L2") as i:
				y[i] = t[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	s.parallelize("L2", "threadIdx.x")
	ast = ir.lower(s.ast())
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	assert "cudaMalloc" in code
	assert "cudaFree" in code
	x_np = np.array([1, 2, 3, 4], dtype="int32")
	y_np = np.zeros((4,), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([3, 5, 7, 9], dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_intrinsic():
	with ir.VarDef([
			("x", (4,), "float32", "input", "gpuglobal"),
			("y", (4,), "float32", "output", "gpuglobal")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = ir.intrinsic("sinf(%)", x[i])

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast())
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([1, 2, 3, 4], dtype="float32")
	y_np = np.zeros((4,), dtype="float32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array(np.sin(x_np), dtype="float32")
	assert np.all(np.isclose(y_np, y_std))

