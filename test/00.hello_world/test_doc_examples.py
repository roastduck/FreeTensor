import freetensor
import pytest


def test_vector_add():
    # Used in docs/index.md and docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 4

    # Change this line to ft.optimize(verbose=1) to see the resulting native code
    @ft.optimize
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
        y = ft.empty((n,), "int32")
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_vector_add_dynamic_length():
    # Used in docs/index.md and docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    @ft.optimize
    def test(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = ft.empty((n,), "int32")
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array(4, dtype="int32"), np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


@pytest.mark.skipif(not freetensor.with_cuda(), reason="requires CUDA")
def test_vector_add_gpu():
    # Used in docs/guide/gpu.md

    import freetensor as ft
    import numpy as np

    # Using the 0-th GPU device
    with ft.GPU(0):

        n = 4

        # Add verbose=1 to see the resulting native code
        @ft.optimize(
            # Parallel Loop Li as GPU threads
            schedule_callback=lambda s: s.parallelize('Li', 'threadIdx.x'))
        def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
            y = ft.empty((n,), "int32")
            #! label: Li # Label the loop below as "Li"
            for i in range(n):
                y[i] = a[i] + b[i]
            return y

        y = test(np.array([1, 2, 3, 4], dtype="int32"),
                 np.array([2, 3, 4, 5], dtype="int32")).numpy()
        print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


@pytest.mark.skipif(not freetensor.with_cuda(), reason="requires CUDA")
def test_vector_add_dynamic_gpu():
    # Used in docs/index.md and docs/guide/gpu.md

    import freetensor as ft
    import numpy as np

    # Using the 0-th GPU device
    with ft.GPU(0):

        @ft.optimize(
            # Parallel Loop Li as GPU threads
            schedule_callback=lambda s: s.parallelize("Li", "threadIdx.x"))
        # Use "byvalue" for `n` so it can be used both during kernel launching
        # and inside a kernel
        def test(n: ft.Var[(), "int32", "input", "byvalue"], a, b):
            a: ft.Var[(n,), "int32"]
            b: ft.Var[(n,), "int32"]
            y = ft.empty((n,), "int32")
            #! label: Li # Label the loop below as "Li"
            for i in range(n):
                y[i] = a[i] + b[i]
            return y

        y = test(np.array(4, dtype="int32"),
                 np.array([1, 2, 3, 4], dtype="int32"),
                 np.array([2, 3, 4, 5], dtype="int32")).numpy()
        print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_vector_add_libop():
    # Used in docs/index.md

    import freetensor as ft
    import numpy as np

    @ft.optimize
    def test(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = a + b  # Or y = ft.add(a, b)
        return y

    y = test(np.array(4, dtype="int32"), np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_dynamic_and_static():
    # Used in docs/guide/first-program.md

    import freetensor as ft
    import numpy as np

    n = 4

    @ft.optimize
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"],
             c: ft.Var[(n,), "int32"]):
        inputs = [a, b, c]  # Static
        y = ft.empty((n,), "int32")  # Dynamic
        for i in range(n):  # Dyanmic
            y[i] = 0  # Dynamic
            for item in inputs:  # Static
                y[i] += item[i]  # Dynamic
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32"),
             np.array([3, 4, 5, 6], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [6, 9, 12, 15])


def test_parallel_vector_add():
    # Used in docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 4

    # Add verbose=1 to see the resulting native code
    @ft.optimize(schedule_callback=lambda s: s.parallelize('Li', 'openmp')
                )  # <-- 2. Apply the schedule
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
        y = ft.empty((n,), "int32")
        #! label: Li  # <-- 1. Label the loop as Li
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_split_and_parallel_vector_add():
    # Used in docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 1024

    def sch(s):
        outer, inner = s.split('Li', 32)
        s.parallelize(outer, 'openmp')

    # Set verbose=1 to see the resulting native code
    # Set verbose=2 to see the code after EVERY schedule
    @ft.optimize(schedule_callback=sch)
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
        y = ft.empty((n,), "int32")
        #! label: Li
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array(np.arange(1024), dtype="int32"),
             np.array(np.arange(1024), dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, np.arange(0, 2048, 2))


def test_grad():
    # Used in docs/index.md and docs/guide/ad.md

    import freetensor as ft
    import numpy as np

    n = 4

    def test(a: ft.Var[(n,), "float32"], b: ft.Var[(n,), "float32"]):
        y = ft.zeros((), "float32")
        for i in range(n):
            y[()] += a[i] * b[i]
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ['a', 'b'],
                                                  [ft.Return()])
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)

    a = np.array([0, 1, 2, 3], dtype="float32")
    b = np.array([3, 2, 1, 0], dtype="float32")
    y = fwd(a, b)
    print(y.numpy())
    dzdy = np.array(1, dtype='float32')
    dzda, dzdb = bwd(**{output_grads[ft.Return()]: dzdy})[input_grads['a'],
                                                          input_grads['b']]
    print(dzda.numpy())
    print(dzdb.numpy())

    assert y.numpy() == 4
    assert np.array_equal(dzda.numpy(), [3, 2, 1, 0])
    assert np.array_equal(dzdb.numpy(), [0, 1, 2, 3])


@pytest.mark.skipif(not freetensor.with_pytorch(), reason="requires PyTorch")
def test_auto_grad_of_softmax():
    # Used docs/guide/ad.md

    import freetensor as ft
    import torch

    n = 4

    def test(x: ft.Var[(n,), "float32"]):
        # Automatically decide gradients for this statement
        m = ft.reduce_max(x, axes=[-1])
        e = ft.exp(x - m)
        s = ft.reduce_sum(e, axes=[-1])
        y = e / s
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ['x'], [ft.Return()])
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)  # Set verbose=1 to see the code

    # Check forward result
    x = torch.rand(n, dtype=torch.float32)
    x.requires_grad = True
    y_ft = fwd(x).torch()
    y_torch = torch.softmax(x, axis=-1)
    assert torch.all(torch.isclose(y_ft, y_torch))

    # Check backward result
    y_torch.grad = dzdy = torch.rand(n, dtype=torch.float32)
    dzdx_ft = bwd(**{output_grads[ft.Return()]: dzdy}).torch()
    y_torch.backward(y_torch.grad)
    dzdx_torch = x.grad
    assert torch.all(torch.isclose(dzdx_ft, dzdx_torch, 1e-4, 1e-7))


@pytest.mark.skipif(not freetensor.with_pytorch(), reason="requires PyTorch")
def test_custom_grad_of_softmax():
    # Used docs/guide/ad.md

    import freetensor as ft
    import torch

    n = 4

    def test(x: ft.Var[(n,), "float32"]):
        # Mark the range that you want to provide graident for, with `StmtRange`
        with ft.StmtRange() as rng:
            m = ft.reduce_max(x, axes=[-1])
            e = ft.exp(x - m)
            s = ft.reduce_sum(e, axes=[-1])
            y = e / s

            # Call `push_for_backward` so we can use forward values in backward
            e_now = ft.push_for_backward(e)
            s_now = ft.push_for_backward(s)
            y_now = ft.push_for_backward(y)
        # Define gradient in `UserGrad`
        with ft.UserGrad(x, y, stmt_range=rng) as (dzdx, dzdy):
            # Retrieve forward value from `y_now`, NOT `y`
            dzds = -ft.reduce_sum(dzdy * y_now, axes=[-1]) / s_now
            dzde = dzdy / s_now + dzds
            dzdx[...] += dzde * e_now  # Use `+=` here
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ['x'], [ft.Return()])
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)  # Set verbose=1 to see the code

    # Check forward result
    x = torch.rand(n, dtype=torch.float32)
    x.requires_grad = True
    y_ft = fwd(x).torch()
    y_torch = torch.softmax(x, axis=-1)
    assert torch.all(torch.isclose(y_ft, y_torch))

    # Check backward result
    y_torch.grad = dzdy = torch.rand(n, dtype=torch.float32)
    dzdx_ft = bwd(**{output_grads[ft.Return()]: dzdy}).torch()
    y_torch.backward(y_torch.grad)
    dzdx_torch = x.grad
    assert torch.all(torch.isclose(dzdx_ft, dzdx_torch, 1e-4, 1e-7))


@pytest.mark.skipif(not freetensor.with_pytorch(), reason="requires PyTorch")
def test_custom_grad_of_softmax_loop_form():
    # Used docs/guide/ad.md

    import freetensor as ft
    import torch

    n = 4

    def test(x: ft.Var[(n, n), "float32"]):
        y = ft.empty((n, n), "float32")
        for i in range(n):
            # Mark the range that you want to provide graident for, with `StmtRange`
            with ft.StmtRange() as rng:
                # `m`, `e` and `s` are local to `i`
                m = ft.reduce_max(x[i], axes=[-1])
                e = ft.exp(x[i] - m)
                s = ft.reduce_sum(e, axes=[-1])
                y[i] = e / s

                # Call `push_for_backward` so we can use forward values in backward
                e_now = ft.push_for_backward(e)
                s_now = ft.push_for_backward(s)
                y_now = ft.push_for_backward(y)
            # Define gradient in `UserGrad`
            with ft.UserGrad(x, y, stmt_range=rng) as (dzdx, dzdy):
                # Retrieve forward value from `y_now`, NOT `y`
                dzds = -ft.reduce_sum(dzdy[i] * y_now[i], axes=[-1]) / s_now
                dzde = dzdy[i] / s_now + dzds
                dzdx[i] += dzde * e_now  # Use `+=` here
        return y

    fwd, bwd, input_grads, output_grads = ft.grad(test, ['x'], [ft.Return()])
    fwd = ft.optimize(fwd)
    bwd = ft.optimize(bwd)  # Set verbose=1 to see the code

    # Check forward result
    x = torch.rand(n, n, dtype=torch.float32)
    x.requires_grad = True
    y_ft = fwd(x).torch()
    y_torch = torch.softmax(x, axis=-1)
    assert torch.all(torch.isclose(y_ft, y_torch))

    # Check backward result
    y_torch.grad = dzdy = torch.rand(n, n, dtype=torch.float32)
    dzdx_ft = bwd(**{output_grads[ft.Return()]: dzdy}).torch()
    y_torch.backward(y_torch.grad)
    dzdx_torch = x.grad
    assert torch.all(torch.isclose(dzdx_ft, dzdx_torch, 1e-4, 1e-7))


@pytest.mark.skipif(not freetensor.with_pytorch(), reason="requires PyTorch")
def test_vector_add_pytorch_io():
    # Used in docs/guide/first-program.md

    import freetensor as ft
    import torch

    n = 4

    # Change this line to ft.optimize(verbose=1) to see the resulting native code
    @ft.optimize
    def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
        y = ft.empty((n,), "int32")
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(torch.tensor([1, 2, 3, 4], dtype=torch.int32),
             torch.tensor([2, 3, 4, 5], dtype=torch.int32)).torch()
    print(y)

    assert torch.all(y == torch.tensor([3, 5, 7, 9], dtype=torch.int32))


@pytest.mark.skipif(not freetensor.with_pytorch(), reason="requires PyTorch")
def test_vector_add_pytorch_function_integration():
    # Used in docs/guide/first-program.md

    import freetensor as ft
    import torch

    n = 4

    # Change this line to ft.optimize_to_pytorch(verbose=1) to see the resulting
    # native code
    @ft.optimize_to_pytorch
    def test(a: ft.Var[(n,), "float32"], b: ft.Var[(n,), "float32"]):
        y = ft.empty((n,), "float32")
        for i in range(n):
            y[i] = a[i] * b[i]
        return y

    # Forward
    a = torch.tensor([1, 2, 3, 4], requires_grad=True, dtype=torch.float32)
    b = torch.tensor([2, 3, 4, 5], requires_grad=True, dtype=torch.float32)
    y = test(a, b)
    print("y = ", y)

    assert torch.all(
        torch.isclose(y, torch.tensor([2, 6, 12, 20], dtype=torch.float32)))

    # Backward
    y.grad = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    y.backward(y.grad)
    print("a.grad = ", a.grad)
    print("b.grad = ", b.grad)

    assert torch.all(
        torch.isclose(a.grad, torch.tensor([2, 3, 4, 5], dtype=torch.float32)))
    assert torch.all(
        torch.isclose(b.grad, torch.tensor([1, 2, 3, 4], dtype=torch.float32)))


def test_sign_hint():
    # Used in docs/guide/hint.md

    import freetensor as ft

    print("Without hint")

    @ft.optimize(verbose=1)  # `verbose=1` prints the code
    def test_no_hint(n: ft.Var[(), "int32"], m: ft.Var[(), "int32"]):
        y = ft.empty((n, m), "int32")
        for i in range(n * m):
            y[i // m, i % m] = i
        return y

    # You will find `runtime_mod` in the code, which involves additional branching
    assert "runtime_mod" in test_no_hint.native_code()
    assert "%" not in test_no_hint.native_code()

    print("With hint")

    @ft.optimize(verbose=1)  # `verbose=1` prints the code
    def test_hint(n: ft.Var[(), "int32"], m: ft.Var[(), "int32>=0"]):
        y = ft.empty((n, m), "int32")
        for i in range(n * m):
            y[i // m, i % m] = i
        return y

    # You will find native C++ `%` in the code, which compiles directly to mod
    # instructions
    assert "runtime_mod" not in test_hint.native_code()
    assert "%" in test_hint.native_code()


def test_assert_hint():
    # Used in docs/guide/hint.md

    import freetensor as ft
    import re

    def sch(s):
        outer, inner = s.split('Li', 32)
        s.parallelize(outer, 'openmp')

    @ft.optimize(schedule_callback=sch, verbose=1)
    def test_no_hint(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = ft.empty((n,), "int32")
        #! label: Li
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    # You will not find a 32-length loop
    assert not re.search(r".* = 0; .* < 32; .*\+\+", test_no_hint.native_code())

    @ft.optimize(schedule_callback=sch, verbose=1)
    def test_hint(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = ft.empty((n,), "int32")
        assert n % 32 == 0
        #! label: Li
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    # You will find a 32-length loop
    assert re.search(r".* = 0; .* < 32; .*\+\+", test_hint.native_code())
