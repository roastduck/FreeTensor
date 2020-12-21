import ir
import numpy as np

def test_hello_world():
	with ir.VarDef("x", (4, 4), ir.DataType.Float32, ir.AccessType.Output) as x:
		x[2, 3] = 2.0
		x[1, 0] = 3.0

	ast = ir.get_ast()
	code, params = ir.code_gen_c(ast)

	x = np.zeros((4, 4), dtype="float32")
	driver = ir.Driver()
	driver.build_and_load(code, len(params))
	driver.set_param(0, x)
	driver.run()

	x_std = np.zeros((4, 4), dtype="float32")
	x_std[2, 3] = 2.0
	x_std[1, 0] = 3.0
	assert np.array_equal(x, x_std)

