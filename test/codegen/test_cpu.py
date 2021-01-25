import ir
import numpy as np

def test_omp_for():
	with ir.VarDef([
			("x", (4,), "int32", "input", "cpu"),
			("y", (4,), "int32", "output", "cpu")]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "openmp")
	ast = ir.lower(s.ast(), ir.CPU())
	print(ast)
	code, params = ir.codegen(ast, ir.CPU())
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

