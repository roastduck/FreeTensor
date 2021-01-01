from ffi import codeGenC as code_gen_c
from ffi import simplifyPass as simplify_pass
from ffi import flattenStmtSeq as flatten_stmt_seq
from ffi import sinkVar as sink_var

def lower(ast):
	ast = simplify_pass(ast)
	ast = sink_var(ast)
	ast = flatten_stmt_seq(ast)
	return ast

