from typing import Optional

import ffi

def lower(ast, target: Optional[ffi.Target]=None):
	ast = ffi.simplify_pass(ast)
	ast = ffi.sink_var(ast)
	ast = ffi.shrink_var(ast)
	ast = ffi.shrink_for(ast)
	ast = ffi.flatten_stmt_seq(ast)
	ast = ffi.merge_if(ast)

	if target is None:
		return ast

	if target.type() == ffi.TargetType.GPU:
		ast = ffi.gpu_make_sync(ast)
		ast = ffi.gpu_correct_shared(ast) # NOTE: No more shrink_var after this pass

	return ast

