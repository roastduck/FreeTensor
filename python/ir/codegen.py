import ffi

def codegen(ast, target: ffi.Target):
	if target.type() == ffi.TargetType.CPU:
		return ffi.code_gen_cpu(ast)
	else:
		assert False, "Unrecognized target %s" % target

