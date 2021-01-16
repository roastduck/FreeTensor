import ffi

def codegen(ast, target: ffi.Target):
	if target.type() == ffi.TargetType.CPU:
		return ffi.code_gen_cpu(ast)
	elif target.type() == ffi.TargetType.GPU:
		return ffi.code_gen_cuda(ast)
	else:
		assert False, "Unrecognized target %s" % target

