import ir
import numpy as np

def test_omp_for():
	with ir.VarDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4, nid="L1") as i:
			y[i] = x[i] + 1

	s = ir.Schedule(ir.pop_ast())
	s.parallelize("L1", "openmp")
	ast = ir.lower(s.ast())
	print(ast)
	code, params = ir.codegen(ast, "cpu")
	print(code)
	x = np.array([1, 2, 3, 4], dtype="int32")
	y = np.zeros((4,), dtype="int32")
	driver = ir.Driver(code, params)
	driver.set_params({"x": x, "y": y})
	driver.run()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y, y_std)

