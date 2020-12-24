from ffi import codeGenC as code_gen_c
from ffi import simplifyPass as simplify_pass

def lower(ast):
	ast = simplify_pass(ast)
	return ast

