import ast
import numpy as np
import sourceinspect as ins

from .nodes import _VarDef, Var, pop_ast

in_transformation = False

def declare_var(var, shape, dtype, atype, mtype):
    pass

def create_var(shape, dtype, atype, mtype):
    return np.zeros(shape, dtype)

class ASTContext:
    def __init__(self):
        self.vardef_stack = []
        self.var_dict = {}
        self.old_vars = []


class ASTContextStack:
    def __init__(self):
        self.ctx_stack = []
        self.now_var_id = {}
        self.name_set = set()
        self.next_nid = ''

    def clear(self):
        self.ctx_stack = []
        self.now_var_id = {}
        self.name_set = set()
        self.next_nid = ''

    def top(self) -> ASTContext:
        return self.ctx_stack[-1]

    def get_current_name(self, name):
        name_id = self.now_var_id.get(name)
        assert name_id is not None, "Variable not found"
        if name_id != 0:
            return '___cache_' + name + '_' + str(name_id)
        return name

    def create_current_name(self, name, atype):
        if atype != "cache":
            if name in self.name_set:
                assert False, "Non-cache variables cannot be redefined"
            self.name_set.add(name)
            self.now_var_id[name] = 0
            return name
        name_id = self.now_var_id.get(name)
        if name_id is None:
            name_id = 1
        else:
            name_id += 1
        while '___cache_' + name + '_' + str(name_id) in self.name_set:
            name_id += 1
        self.now_var_id[name] = name_id
        if name_id:
            name = '___cache_' + name + '_' + str(name_id)
        self.name_set.add(name)
        return name

    def find_var_by_name(self, name):
        name = self.get_current_name(name)

        for ctx in reversed(self.ctx_stack):  # type: ASTContext
            if name in ctx.old_vars:
                assert False, "Variable reassigned in if/for/while"
            var = ctx.var_dict.get(name)
            if var is not None:
                return var

        assert False, "Bug: variable not found by find_var_by_name"

    def create_scope(self):
        self.ctx_stack.append(ASTContext())

    def pop_scope(self):
        assert self.ctx_stack, "Bug: scope stack is empty when pop_scope"
        popped = self.ctx_stack.pop()  # type: ASTContext
        if self.ctx_stack:
            top = self.top()
            top.old_vars.extend(popped.var_dict.keys())
        for var in reversed(popped.vardef_stack):  # type: _VarDef
            var.__exit__(None, None, None)

    def create_variable(self, name, shape, dtype, atype, mtype):
        name = self.create_current_name(name, atype)
        vardef = _VarDef(name, shape, dtype, atype, mtype)
        var = vardef.__enter__()
        top = self.top()
        top.vardef_stack.append(vardef)
        top.var_dict[name] = var
        return var

    def set_nid(self, name):
        self.next_nid = name

    def get_nid(self):
        ret = self.next_nid
        self.next_nid = ''
        return ret

ctx_stack = ASTContextStack()


class ASTTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_stmt(stmt):
        return ast.parse(stmt).body[0]

    @staticmethod
    def parse_expr(expr):
        return ast.parse(expr).body[0].value

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.args.args = []
        prologue = [self.parse_stmt('ir.transformer.ctx_stack.create_scope()')]
        epilogue = [self.parse_stmt('ir.transformer.ctx_stack.pop_scope()')]
        node.body = [self.parse_stmt('import ir')] + prologue + node.body + epilogue
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        print(ast.dump(node))
        # TODO: (maybe) support for multiple assignment
        assert len(node.targets) == 1, "Multiple assignment is not supported"
        if isinstance(node.value, ast.Call) and \
              isinstance(node.value.func, ast.Attribute) and \
              isinstance(node.value.func.value, ast.Name) and \
              node.value.func.value.id == 'ir' and \
              node.value.func.attr == 'create_var':
            name = node.targets[0].id
            template = "{} = ir.transformer.ctx_stack.create_variable(0, 0, 0, 0, 0)".format(name)
            new_node = self.parse_stmt(template)
            new_node.value.args = [self.parse_expr('"' + name + '"')] + node.value.args
            return new_node
        return node

    def visit_For(self, node):
        nid = ctx_stack.get_nid()
        self.generic_visit(node)
        if isinstance(node.iter, ast.Call) and \
              isinstance(node.iter.func, ast.Name) and \
              node.iter.func.id == 'range' and \
              len(node.iter.args) == 2:
            name = node.target.id
            template = '''
if 1:
    ir.transformer.ctx_stack.create_scope()
    with ir.For(0, 0, 0, 0) as {}:
        pass
    ir.transformer.ctx_stack.pop_scope()
            '''.format(name)
            new_node = self.parse_stmt(template)
            wt = new_node.body[1]
            args = [self.parse_expr('"' + name + '"')] + node.iter.args
            args.append(self.parse_expr('"' + nid + '"'))
            wt.items[0].context_expr.args = args
            wt.body = node.body
            return new_node
        else:
            assert False, "For statement other than range(a, b) is not implemented"

    def visit_Expr(self, node):
        self.generic_visit(node)
        print(ast.dump(node))
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            s = node.value.value
            if s[0:9] == 'for-nid: ':
                ctx_stack.set_nid(s[9:])
        elif isinstance(node.value, ast.Call) and \
              isinstance(node.value.func, ast.Attribute) and \
              isinstance(node.value.func.value, ast.Name) and \
              node.value.func.value.id == 'ir' and \
              node.value.func.attr == 'declare_var':
            name = node.value.args[0].id
            template = "{} = ir.transformer.ctx_stack.create_variable(0, 0, 0, 0, 0)".format(name)
            new_node = self.parse_stmt(template)
            new_node.value.args = node.value.args
            new_node.value.args[0] = self.parse_expr('"' + name + '"')
            print(ast.dump(new_node))
            return new_node
        return node

    def visit_If(self, node):
        template = '''
if 1:
    ir.transformer.ctx_stack.create_scope()
    with ir.If(0):
        pass
    with ir.Else():
        pass
    ir.transformer.ctx_stack.pop_scope()
        '''
        new_node = self.parse_stmt(template)
        wif = new_node.body[1]
        wif.items[0].context_expr.args = [node.test]
        wif.body = node.body
        new_node.body[2].body = node.orelse

        return new_node

def _get_global_vars(func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    import copy
    global_vars = copy.copy(func.__globals__)

    freevar_names = func.__code__.co_freevars
    closure = func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


def remove_indent(lines):
    lines = lines.split('\n')
    to_remove = 0
    for i in range(len(lines[0])):
        if lines[0][i] == ' ':
            to_remove = i + 1
        else:
            break

    cleaned = []
    for l in lines:
        cleaned.append(l[to_remove:])
        if len(l) >= to_remove:
            for i in range(to_remove):
                assert l[i] == ' '

    return '\n'.join(cleaned)


def transform(func):
    ctx_stack.clear()
    src = remove_indent(ins.getsource(func))
    tree = ast.parse(src)
    ASTTransformer().visit(tree)
    tree = ast.fix_missing_locations(tree)
    local_vars = {}
    global_vars = _get_global_vars(func)
    print(ast.dump(tree))
    exec(
        compile(tree,
                filename=ins.getsourcefile(func),
                mode='exec'), global_vars, local_vars)
    compiled = local_vars[func.__name__]
    compiled()
    return pop_ast()
