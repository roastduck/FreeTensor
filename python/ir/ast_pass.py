from ffi import codeGenC as code_gen_c
from ffi import simplifyPass as simplify_pass
from ffi import flattenStmtSeq as flatten_stmt_seq

def lower(ast):
	ast = simplify_pass(ast)
	ast = flatten_stmt_seq(ast)
	return ast

