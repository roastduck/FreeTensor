import ir
import pytest
import inspect


# DO NOT ADD LINES IN THIS FUNCTION
def test_assert_in_ctx_stack():

    @ir.inline
    def bar():
        c = ir.create_var((1,), "int32", "cpu")
        for i in range(1):
            c = ir.create_var((1,), "int32", "cpu")
        c[0]

    @ir.inline
    def foo():
        bar()

    with pytest.raises(ir.InvalidProgram) as e:

        @ir.transform
        def test():
            foo()

    frame_info = inspect.getframeinfo(inspect.currentframe())
    line_foo = frame_info.lineno - 2
    line_bar = frame_info.lineno - 8
    line_c = frame_info.lineno - 12
    file = frame_info.filename

    msg = f"""
On line {line_foo} in file {file}: 
            foo()
On line {line_bar} in file {file}: 
        bar()
On line {line_c} in file {file}: 
        c[0]
"""
    print(e.value.args[0])
    assert e.value.args[0][:len(msg)] == msg


# DO NOT ADD LINES IN THIS FUNCTION
def test_assert_in_transformer():

    @ir.inline
    def bar(a, b):
        a @ b

    @ir.inline
    def foo(a, b):
        bar(a, b)

    with pytest.raises(ir.InvalidProgram) as e:

        @ir.transform
        def test(a, b):
            ir.declare_var(a, (1,), "int32", "input", "cpu")
            ir.declare_var(b, (1,), "int32", "input", "cpu")
            foo(a, b)

    frame_info = inspect.getframeinfo(inspect.currentframe())
    line_foo = frame_info.lineno - 2
    line_bar = frame_info.lineno - 10
    line_ab = frame_info.lineno - 14
    file = frame_info.filename

    msg = f"""
On line {line_foo} in file {file}: 
            foo(a, b)
On line {line_bar} in file {file}: 
        bar(a, b)
On line {line_ab} in file {file}: 
        a @ b
AssertionError: Binary operator not implemented"""
    print(e.value.args[0])
    assert e.value.args[0] == msg
