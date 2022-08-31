import freetensor as ft
import numpy as np
import torch
import pytest


def test_numpy_strided():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int32"]):
        y = ft.empty((2, 2), "int32")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = np.array([[0, 1], [2, 3]], dtype="int32").transpose()
    assert x.strides[0] < x.strides[1]
    y = test(x).numpy()
    assert np.array_equal(y, np.array([[1, 3], [2, 4]], dtype="int32"))


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_torch():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int32"]):
        y = ft.empty((2, 2), "int32")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    y = test(x).torch()
    assert torch.all(y == torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_torch_strided():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int32"]):
        y = ft.empty((2, 2), "int32")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).transpose(1, 0)
    assert not x.is_contiguous()
    y = test(x).torch()
    assert torch.all(y == torch.tensor([[1, 3], [2, 4]], dtype=torch.int32))


@pytest.mark.skipif(not ft.with_pytorch() or not ft.with_cuda(),
                    reason="requires PyTorch and CUDA")
def test_torch_cuda():

    with ft.GPU():

        def sch(s):
            s.parallelize("Li", "blockIdx.x")
            s.parallelize("Lj", "threadIdx.x")

        @ft.optimize(schedule_callback=sch)
        def test(x: ft.Var[(2, 2), "int32"]):
            y = ft.empty((2, 2), "int32")
            #! label: Li
            for i in range(2):
                #! label: Lj
                for j in range(2):
                    y[i, j] = x[i, j] + 1
            return y

        x = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).cuda()
        y = test(x).torch()
        assert y.is_cuda
        assert torch.all(
            y == torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).cuda())
