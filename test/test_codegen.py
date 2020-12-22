import ir
import numpy as np

def test_hello_world():
	with ir.VarDef("x", (4, 4), ir.DataType.Float32, ir.AccessType.Output) as x:
		x[2, 3] = 2.0
		x[1, 0] = 3.0

	ast = ir.pop_ast()
	code, params = ir.code_gen_c(ast)

	x = np.zeros((4, 4), dtype="float32")
	driver = ir.Driver(params)
	driver.build_and_load(code)
	driver.set_params({"x": x})
	driver.run()

	x_std = np.zeros((4, 4), dtype="float32")
	x_std[2, 3] = 2.0
	x_std[1, 0] = 3.0
	assert np.array_equal(x, x_std)

def test_scalar_op():
	with ir.VarsDef([
			("x", (), ir.DataType.Int32, ir.AccessType.Input),
			("y", (), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		y[()] = x[()] * 2 + 1

	code, params = ir.code_gen_c(ir.pop_ast())
	x = np.array(5, dtype="int32")
	y = np.array(0, dtype="int32")
	driver = ir.Driver(params)
	driver.build_and_load(code)
	driver.set_params({"x": x, "y": y})
	driver.run()

	assert y[()] == 11

def test_for():
	with ir.VarsDef([
			("x", (4,), ir.DataType.Int32, ir.AccessType.Input),
			("y", (4,), ir.DataType.Int32, ir.AccessType.Output)]) as (x, y):
		with ir.For("i", 0, 4) as i:
			y[i] = x[i] + 1

	code, params = ir.code_gen_c(ir.pop_ast())
	x = np.array([1, 2, 3, 4], dtype="int32")
	y = np.zeros((4,), dtype="int32")
	driver = ir.Driver(params)
	driver.build_and_load(code)
	driver.set_params({"x": x, "y": y})
	driver.run()

	y_std = np.array([2, 3, 4, 5], dtype="int32")
	assert np.array_equal(y, y_std)

