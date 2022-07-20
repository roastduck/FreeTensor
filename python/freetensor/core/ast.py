import freetensor_ffi as ffi
from . import passes
from . import driver

def dump_ast(ast, dtype_in_load=False):
    return ffi.dump_ast(ast, dtype_in_load)

def load_ast(ast_str):
    return ffi.load_ast(ast_str)

def remote_lower_ast(ast_str, target_str):
    return dump_ast(passes.lower(load_ast(ast_str), driver.load_target(target_str)))


