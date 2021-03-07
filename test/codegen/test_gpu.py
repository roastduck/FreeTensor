import ir
import pytest
import numpy as np

target = ir.GPU()
# TODO: Detect GPU arch and set it to target
device = ir.Device(target)

host = ir.Device(ir.CPU())

def test_basic():
	with ir.VarDef([
			("x", (4,), "int32", "input", "gpu/global"),
			("y", (4,), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
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
			("x", (4,), "int32", "input", "gpu/global"),
			("y", (4,), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			ir.MarkNid("S1")
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.cache("S1", "x", "gpu/shared")
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
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
			("x", (4,), "int32", "input", "gpu/global"),
			("y", (4,), "int32", "output", "gpu/global")]) as (x, y):
		with ir.VarDef("t", (4,), "int32", "cache", "gpu/global") as t:
			with ir.For("i", 0, 4, nid="L1") as i:
				t[i] = x[i] * 2
			with ir.For("i", 0, 4, nid="L2") as i:
				y[i] = t[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	s.parallelize("L2", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
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

def test_pass_by_value_0d():
	with ir.VarDef("n", (), "int32", "input", "byvalue") as n:
		with ir.VarDef([
				("x", (n[()], 4), "int32", "input", "gpu/global"),
				("y", (n[()], 4), "int32", "output", "gpu/global")]) as (x, y):
			with ir.For("i", 0, 4, nid="L1") as i:
				with ir.For("j", 0, n[()], nid="L2") as j:
					y[j, i] = x[j, i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	n_np = np.array(5, dtype="int32")
	x_np = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
	y_np = np.zeros((5, 4), dtype="int32")
	n_arr = ir.Array(n_np, host)
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"n": n_arr, "x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([[2, 3, 4, 5]] * 5, dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_pass_by_value_1d():
	with ir.VarDef("n", (1,), "int32", "input", "byvalue") as n:
		with ir.VarDef([
				("x", (n[0], 4), "int32", "input", "gpu/global"),
				("y", (n[0], 4), "int32", "output", "gpu/global")]) as (x, y):
			with ir.For("i", 0, 4, nid="L1") as i:
				with ir.For("j", 0, n[0], nid="L2") as j:
					y[j, i] = x[j, i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)
	code, params = ir.codegen(ast, target)
	print(code)
	n_np = np.array([5], dtype="int32")
	x_np = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
	y_np = np.zeros((5, 4), dtype="int32")
	n_arr = ir.Array(n_np, host)
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"n": n_arr, "x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([[2, 3, 4, 5]] * 5, dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_intrinsic():
	with ir.VarDef([
			("x", (4,), "float32", "input", "gpu/global"),
			("y", (4,), "float32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = ir.intrinsic("sinf(%)", x[i])

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
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

def test_syncthreads():
	with ir.VarDef([
			("x", (4, 256), "int32", "input", "gpu/global"),
			("y", (4, 256), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L0") as i:
			with ir.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
				with ir.For("j", 0, 256, nid="L1") as j:
					t[j] = x[i, j] * 2
				with ir.For("j", 0, 256, nid="L2") as j:
					y[i, j] = t[255 - j] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L0", "blockIdx.x")
	s.parallelize("L1", "threadIdx.x")
	s.parallelize("L2", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	with ir.VarDef([
			("x", (4, 256), "int32", "input", "gpu/global"),
			("y", (4, 256), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For(".blockIdx.x", 0, 4) as i:
			with ir.For(".threadIdx.x", 0, 256) as j:
				with ir.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
					ir.Any()
					ir.Eval(ir.intrinsic("__syncthreads()"))
					ir.Any()
					ir.Eval(ir.intrinsic("__syncthreads()"))
	assert ir.pop_ast().match(ast)

	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([range(256)] * 4, dtype="int32")
	y_np = np.zeros((4, 256), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([range(511, -1, -2)] * 4, dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_syncwarp():
	with ir.VarDef([
			("x", (4, 4), "int32", "input", "gpu/global"),
			("y", (4, 4), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L0") as i:
			with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
				with ir.For("j", 0, 4, nid="L1") as j:
					t[j] = x[i, j] * 2
				with ir.For("j", 0, 4, nid="L2") as j:
					y[i, j] = t[3 - j] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L0", "blockIdx.x")
	s.parallelize("L1", "threadIdx.x")
	s.parallelize("L2", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	with ir.VarDef([
			("x", (4, 4), "int32", "input", "gpu/global"),
			("y", (4, 4), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For(".blockIdx.x", 0, 4) as i:
			with ir.For(".threadIdx.x", 0, 4) as j:
				with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
					ir.Any()
					ir.Eval(ir.intrinsic("__syncwarp()"))
					ir.Any()
					ir.Eval(ir.intrinsic("__syncwarp()"))
	assert ir.pop_ast().match(ast)

	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([[0, 1, 2, 3]] * 4, dtype="int32")
	y_np = np.zeros((4, 4), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([[7, 5, 3, 1]] * 4, dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_correct_shared():
	with ir.VarDef([
			("x", (4, 256), "int32", "input", "gpu/global"),
			("y", (4, 256), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L0") as i:
			with ir.VarDef("t", (256,), "int32", "cache", "gpu/shared") as t:
				with ir.For("j", 0, 256, nid="L1") as j:
					t[j] = x[i, j] * 2
				with ir.For("j", 0, 256, nid="L2") as j:
					y[i, j] = t[j] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L0", "threadIdx.y")
	s.parallelize("L1", "threadIdx.x")
	s.parallelize("L2", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	with ir.VarDef([
			("x", (4, 256), "int32", "input", "gpu/global"),
			("y", (4, 256), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For(".threadIdx.y", 0, 4) as i:
			with ir.For(".threadIdx.x", 0, 256) as j:
				with ir.VarDef("t", (4, 256), "int32", "cache", "gpu/shared") as t:
					t[i, j] = x[i, j] * 2
					ir.Eval(ir.intrinsic("__syncthreads()"))
					y[i, j] = t[i, j] + 1
					ir.Eval(ir.intrinsic("__syncthreads()"))
	assert ir.pop_ast().match(ast)

	code, params = ir.codegen(ast, target)
	print(code)
	x_np = np.array([range(256)] * 4, dtype="int32")
	y_np = np.zeros((4, 256), dtype="int32")
	x_arr = ir.Array(x_np, device)
	y_arr = ir.Array(y_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"x": x_arr, "y": y_arr})
	driver.run()
	y_np = y_arr.numpy()

	y_std = np.array([range(1, 513, 2)] * 4, dtype="int32")
	assert np.array_equal(y_np, y_std)

def test_parallel_different_length():
	with ir.VarDef([
			("a", (4, 4), "int32", "input", "gpu/global"),
			("b", (4, 8), "int32", "input", "gpu/global"),
			("c", (4, 8), "int32", "output", "gpu/global")]) as (a, b, c):
		with ir.For("i", 0, 4, nid="L0") as i:
			with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
				with ir.For("j", 0, 4, nid="L1") as j:
					t[j] = a[i, j]
				with ir.For("j", 0, 4, nid="L2") as j:
					with ir.For("k", 0, 8, nid="L3") as k:
						c[i, k] = c[i, k] + t[j] * b[j, k]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L0", "blockIdx.x")
	s.parallelize("L1", "threadIdx.x")
	s.parallelize("L3", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	with ir.VarDef([
			("a", (4, 4), "int32", "input", "gpu/global"),
			("b", (4, 8), "int32", "input", "gpu/global"),
			("c", (4, 8), "int32", "output", "gpu/global")]) as (a, b, c):
		with ir.For(".blockIdx.x", 0, 4) as blk:
			with ir.For(".threadIdx.x", 0, 8) as th:
				with ir.VarDef("t", (4,), "int32", "cache", "gpu/shared") as t:
					with ir.If(th < 4):
						t[th] = a[blk, th]
						ir.Eval(ir.intrinsic("__syncwarp()"))
					with ir.For("j", 0, 4) as j:
						ir.Any()
						ir.Eval(ir.intrinsic("__syncwarp()"))
	assert ir.pop_ast().match(ast)

	code, params = ir.codegen(ast, target)
	print(code)
	a_np = np.random.rand(4, 4).astype("int32")
	b_np = np.random.rand(4, 8).astype("int32")
	c_np = np.zeros((4, 8), dtype="int32")
	a_arr = ir.Array(a_np, device)
	b_arr = ir.Array(b_np, device)
	c_arr = ir.Array(c_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
	driver.run()
	c_np = c_arr.numpy()

	c_std = a_np @ b_np
	assert np.array_equal(c_np, c_std)

def test_parallel_broadcast():
	with ir.VarDef([
			("a", (4, 1), "int32", "input", "gpu/global"),
			("b", (1, 8), "int32", "input", "gpu/global"),
			("c", (4, 8), "int32", "output", "gpu/global")]) as (a, b, c):
		with ir.For("i", 0, 4, nid="L0") as i:
			with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
				t[0] = a[i, 0]
				with ir.For("k", 0, 8, nid="L1") as k:
					c[i, k] = c[i, k] + t[0] * b[0, k]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L0", "blockIdx.x")
	s.parallelize("L1", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	with ir.VarDef([
			("a", (4, 1), "int32", "input", "gpu/global"),
			("b", (1, 8), "int32", "input", "gpu/global"),
			("c", (4, 8), "int32", "output", "gpu/global")]) as (a, b, c):
		with ir.For(".blockIdx.x", 0, 4) as blk:
			with ir.For(".threadIdx.x", 0, 8) as th:
				with ir.VarDef("t", (1,), "int32", "cache", "gpu/shared") as t:
					with ir.If(th == 0):
						t[0] = a[blk, 0]
					ir.Eval(ir.intrinsic("__syncwarp()"))
					ir.Any()
					ir.Eval(ir.intrinsic("__syncwarp()"))
	assert ir.pop_ast().match(ast)

	code, params = ir.codegen(ast, target)
	print(code)
	a_np = np.random.rand(4, 1).astype("int32")
	b_np = np.random.rand(1, 8).astype("int32")
	c_np = np.zeros((4, 8), dtype="int32")
	a_arr = ir.Array(a_np, device)
	b_arr = ir.Array(b_np, device)
	c_arr = ir.Array(c_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
	driver.run()
	c_np = c_arr.numpy()

	c_std = a_np @ b_np
	assert np.array_equal(c_np, c_std)

def test_unbounded_length():
	with ir.VarDef("n", (), "int32", "input", "gpu/global") as n:
		with ir.VarDef([
				("x", (n[()],), "int32", "input", "gpu/global"),
				("y", (n[()],), "int32", "output", "gpu/global")]) as (x, y):
			with ir.For("i", 0, n[()], nid="L1") as i:
				y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "threadIdx.x")
	with pytest.raises(ir.InvalidProgram):
		print(ir.lower(s.ast(), target))

def test_parallel_reduction():
	with ir.VarDef([
			("x", (4, 64), "int32", "input", "gpu/global"),
			("y", (4,), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 64, nid="L2") as j:
				y[i] = y[i] + x[i, j]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "blockIdx.x")
	s.parallelize("L2", "threadIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	code, params = ir.codegen(ast, target)
	assert "atomicAdd" in code
	assert "+=" not in code
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
			("x", (4, 64), "int32", "input", "gpu/global"),
			("y", (4,), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 64, nid="L2") as j:
				y[i] = y[i] + x[i, j]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "blockIdx.x")
	ast = ir.lower(s.ast(), target)
	print(ast)

	code, params = ir.codegen(ast, target)
	assert "atomicAdd" not in code
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
			("x", (4, 64), "int32", "input", "gpu/global"),
			("y", (4,), "int32", "output", "gpu/global")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			with ir.For("j", 0, 64, nid="L2") as j:
				y[i] = y[i] + x[i, j]

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "blockIdx.x")
	s.unroll("L2")
	ast = ir.lower(s.ast(), target)
	print(ast)

	code, params = ir.codegen(ast, target)
	assert "atomicAdd" not in code
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
