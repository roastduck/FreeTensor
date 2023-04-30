import freetensor as ft
import torch
import pytest


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_basic():

    @ft.optimize_to_pytorch(verbose=1)
    def sinh(x: ft.Var[(4,), "float64"]):
        y = ft.empty((4,), "float64")
        for i in range(4):
            y[i] = (ft.exp(x[i]) - ft.exp(-x[i])) / 2
        return y

    # Test inference without grad
    x = torch.rand(4, requires_grad=False, dtype=torch.double)
    assert torch.all(torch.isclose(sinh(x), (torch.exp(x) - torch.exp(-x)) / 2))

    # Test forward with grad
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.all(torch.isclose(sinh(x), (torch.exp(x) - torch.exp(-x)) / 2))

    # Test backward
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(sinh, x)


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_multiple_funcs():

    @ft.optimize_to_pytorch(verbose=1)
    def exp_neg(x: ft.Var[(), "float64"]):
        y = ft.empty((), "float64")
        y[...] = ft.exp(-x[...])
        return y

    @ft.optimize_to_pytorch(verbose=1)
    def sinh(x: ft.Var[(4,), "float64"]):
        y = ft.empty((4,), "float64")
        for i in range(4):
            y[i] = (ft.exp(x[i]) - exp_neg(x[i])) / 2
        return y

    # Test inference without grad
    x = torch.rand(4, requires_grad=False, dtype=torch.double)
    assert torch.all(torch.isclose(sinh(x), (torch.exp(x) - torch.exp(-x)) / 2))

    # Test forward with grad
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.all(torch.isclose(sinh(x), (torch.exp(x) - torch.exp(-x)) / 2))

    # Test backward
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(sinh, x)


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_return_var_to_be_taped():

    @ft.optimize_to_pytorch(verbose=1)
    def exp(x: ft.Var[(4,), "float64"]):
        y = ft.empty((4,), "float64")
        for i in range(4):
            y[i] = ft.exp(x[i])
        return y

    # Test inference without grad
    x = torch.rand(4, requires_grad=False, dtype=torch.double)
    assert torch.all(torch.isclose(exp(x), torch.exp(x)))

    # Test forward with grad
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.all(torch.isclose(exp(x), torch.exp(x)))

    # Test backward
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(exp, x)
