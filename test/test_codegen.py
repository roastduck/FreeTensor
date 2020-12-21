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
	with ir.VarDef("x", (), ir.DataType.Int32, ir.AccessType.Input) as x:
		with ir.VarDef("y", (), ir.DataType.Int32, ir.AccessType.Output) as y:
			y[()] = x[()] * 2 + 1

	code, params = ir.code_gen_c(ir.pop_ast())
	x = np.array(5, dtype="int32")
	y = np.array(0, dtype="int32")
	driver = ir.Driver(params)
	driver.build_and_load(code)
	driver.set_params({"x": x, "y": y})
	driver.run()

	assert y[()] == 11

