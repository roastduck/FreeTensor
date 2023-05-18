import freetensor as ft
import pytest

if not ft.with_pytorch():
    pytest.skip(
        "The tests requires PyTorch, and FreeTensor is expected to be built with "
        "PyTorch to be compatible with it, even if there is no direct interaction "
        "between FreeTensor and PyTorch",
        allow_module_level=True)

import torch


def test_basic():

    @ft.transform(verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        return ft.libop.matmul(a, b)

    fwd, bwd, input_map = ft.jacrev(f, ["a", "b"], ft.Return(), verbose=1)
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = fwd(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())
    jac = bwd()
    a_jac_torch = torch.tensor(jac[input_map['a']].numpy())
    b_jac_torch = torch.tensor(jac[input_map['b']].numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))

    a_jac_std, b_jac_std = torch.func.jacrev(torch.matmul,
                                             argnums=(0, 1))(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))
    assert torch.all(torch.isclose(b_jac_torch, b_jac_std))


def test_part_of_inputs():

    @ft.transform(verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        return ft.libop.matmul(a, b)

    fwd, bwd, input_map = ft.jacrev(f, ["a"], ft.Return(), verbose=1)
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = fwd(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())
    a_jac_torch = torch.tensor(bwd().numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))

    a_jac_std = torch.func.jacrev(torch.matmul, argnums=0)(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))


def test_part_of_outputs():

    @ft.transform(verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        c = ft.libop.matmul(a, b)
        d = c + 1
        return c, d

    fwd, bwd, input_map = ft.jacrev(f, ["a", "b"], ft.Return(0), verbose=1)
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    c_arr, d_arr = fwd(a_arr, b_arr)
    c_torch = torch.tensor(c_arr.numpy())
    d_torch = torch.tensor(d_arr.numpy())
    jac = bwd()
    a_jac_torch = torch.tensor(jac[input_map['a']].numpy())
    b_jac_torch = torch.tensor(jac[input_map['b']].numpy())

    c_std = torch.matmul(a_torch, b_torch)
    d_std = c_std + 1
    assert torch.all(torch.isclose(c_torch, c_std))
    assert torch.all(torch.isclose(d_torch, d_std))

    a_jac_std, b_jac_std = torch.func.jacrev(torch.matmul,
                                             argnums=(0, 1))(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))
    assert torch.all(torch.isclose(b_jac_torch, b_jac_std))


def test_flatten():

    @ft.transform(verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        return ft.libop.matmul(a, b)

    fwd, bwd, input_map = ft.jacrev(f, ["a", "b"],
                                    ft.Return(),
                                    flatten=True,
                                    verbose=1)
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd, verbose=1)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = fwd(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())
    jac_torch = torch.tensor(bwd().numpy())
    a_jac_torch = jac_torch[:, :20].reshape(4, 6, 4, 5)
    b_jac_torch = jac_torch[:, 20:].reshape(4, 6, 5, 6)

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))

    a_jac_std, b_jac_std = torch.func.jacrev(torch.matmul,
                                             argnums=(0, 1))(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))
    assert torch.all(torch.isclose(b_jac_torch, b_jac_std))


def test_attach_backward():

    @ft.optimize
    @ft.jacrev(inputs=["a", "b"],
               output=ft.Return(),
               attach_backward=True,
               verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        return ft.libop.matmul(a, b)

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = f(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())
    jac = f.backward()
    a_jac_torch = torch.tensor(jac[f.input_name_to_gradient_name['a']].numpy())
    b_jac_torch = torch.tensor(jac[f.input_name_to_gradient_name['b']].numpy())

    y_std = torch.matmul(a_torch, b_torch)
    assert torch.all(torch.isclose(y_torch, y_std))

    a_jac_std, b_jac_std = torch.func.jacrev(torch.matmul,
                                             argnums=(0, 1))(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))
    assert torch.all(torch.isclose(b_jac_torch, b_jac_std))


def test_inline_invoke_func_from_jacrev():

    @ft.jacrev(inputs=["a", "b"],
               output=ft.Return(),
               tapes=ft.GradTapeMode.All,
               attach_backward=True,
               tape_in_closure=False,
               verbose=1)
    def f(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        # Square so we can test for tapes
        return ft.square(ft.libop.matmul(a, b))

    @ft.optimize
    def fwd_then_bwd(a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        fwd_ret = f(a, b)
        y = None
        tapes = {}
        if isinstance(fwd_ret, ft.VarRef):
            y = fwd_ret
        else:
            y = fwd_ret[0]
            for (tape_decl, tape_val) in zip(f.returns[1:], fwd_ret[1:]):
                tapes[tape_decl.name] = tape_val
        jac = f.backward(a, b, **tapes)
        input_map = f.input_name_to_gradient_name
        return y, jac[input_map['a']], jac[input_map['b']]

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr, a_jac_arr, b_jac_arr = fwd_then_bwd(a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())
    a_jac_torch = torch.tensor(a_jac_arr.numpy())
    b_jac_torch = torch.tensor(b_jac_arr.numpy())

    y_std = torch.matmul(a_torch, b_torch)**2
    assert torch.all(torch.isclose(y_torch, y_std))

    a_jac_std, b_jac_std = torch.func.jacrev(lambda a, b: torch.matmul(a, b)**2,
                                             argnums=(0, 1))(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))
    assert torch.all(torch.isclose(b_jac_torch, b_jac_std))
