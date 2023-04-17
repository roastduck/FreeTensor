import freetensor as ft
import numpy as np
import torch
import pytest


def test_build_array_from_list():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int32"]):
        y = ft.empty((2, 2), "int32")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = ft.array([[0, 2], [1, 3]], dtype="int32")
    y = test(x)
    assert np.array_equal(y.numpy(), np.array([[1, 3], [2, 4]], dtype="int32"))


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


def test_numpy_cast():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int64"]):
        y = ft.empty((2, 2), "int64")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = np.array([[0, 2], [1, 3]], dtype="int32")
    x_arr = ft.array(x, dtype="int64")
    y = test(x_arr).numpy()
    assert np.array_equal(y, np.array([[1, 3], [2, 4]], dtype="int64"))


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


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_torch_cast():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int64"]):
        y = ft.empty((2, 2), "int64")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    x_arr = ft.array(x, dtype="int64")
    y = test(x_arr).torch()
    assert torch.all(y == torch.tensor([[1, 3], [2, 4]], dtype=torch.int64))


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


@pytest.mark.skipif(not ft.with_pytorch() or not ft.with_cuda(),
                    reason="requires PyTorch and CUDA")
def test_torch_cuda_auto_copy_read():

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

        x = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).cpu()  # FROM CPU
        y = test(x).torch()
        assert y.is_cuda
        assert torch.all(
            y == torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).cuda())


@pytest.mark.skipif(not ft.with_pytorch() or not ft.with_cuda(),
                    reason="requires PyTorch and CUDA")
def test_torch_cuda_no_write_to_other_device():

    with ft.GPU():

        def sch(s):
            s.parallelize("Li", "blockIdx.x")
            s.parallelize("Lj", "threadIdx.x")

        @ft.optimize(schedule_callback=sch)
        def test(x, y):
            x: ft.Var[(2, 2), "int32", "input"]
            y: ft.Var[(2, 2), "int32", "inout"]
            #! label: Li
            for i in range(2):
                #! label: Lj
                for j in range(2):
                    y[i, j] += x[i, j]

        x = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).cpu()
        y = torch.tensor([[1, 1], [1, 1]], dtype=torch.int32).cpu()  # FROM CPU
        with pytest.raises(ft.InvalidIO):
            test(x, y)


@pytest.mark.skipif(not ft.with_pytorch() or not ft.with_cuda(),
                    reason="requires PyTorch and CUDA")
def test_torch_cuda_explicitly_copy_from_other_device():

    with ft.GPU():

        def sch(s):
            s.parallelize("Li", "blockIdx.x")
            s.parallelize("Lj", "threadIdx.x")

        @ft.optimize(schedule_callback=sch)
        def test(x, y):
            x: ft.Var[(2, 2), "int32", "input"]
            y: ft.Var[(2, 2), "int32", "inout"]
            #! label: Li
            for i in range(2):
                #! label: Lj
                for j in range(2):
                    y[i, j] += x[i, j]

        x = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).cpu()
        y = torch.tensor([[1, 1], [1, 1]], dtype=torch.int32).cpu()  # FROM CPU
        y = ft.array(y)
        test(x, y)
        y = y.torch()
        assert y.is_cuda
        assert torch.all(
            y == torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).cuda())
