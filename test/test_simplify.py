import ir
import numpy as np

def test_const_fold():
	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			y[i] = 0 * i

	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	assert "y[i] = 0" in str(ast)

def test_redundant_if():
	with ir.VarDef("y", (4,), ir.DataType.Int32, ir.AccessType.Output) as y:
		with ir.For("i", 0, 4) as i:
			with ir.If(i < 10):
				y[i] = 1

	ast = ir.pop_ast()
	print(ast)
	ast = ir.lower(ast)
	print(ast)

	assert "if" not in str(ast)

