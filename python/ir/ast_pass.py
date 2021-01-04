from ffi import code_gen_c
from ffi import simplify_pass
from ffi import sink_var
from ffi import shrink_var
from ffi import flatten_stmt_seq

def lower(ast):
	ast = simplify_pass(ast)
	ast = sink_var(ast)
	ast = shrink_var(ast)
	ast = flatten_stmt_seq(ast)
	return ast

