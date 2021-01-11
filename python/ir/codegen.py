from ffi import code_gen_cpu

def codegen(ast, target: str):
	if target == "cpu":
		return code_gen_cpu(ast)
	else:
		assert False, "Unrecognized target %s" % target

