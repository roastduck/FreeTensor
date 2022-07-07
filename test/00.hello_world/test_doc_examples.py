import freetensor
import pytest


def test_vector_add():
    # Used in docs/index.md and docs/guide/schedules.md

    import freetensor as ft
    import numpy as np

    n = 4

    # Change this line to ft.optimize(verbose=1) to see the resulting native code
    @ft.optimize(verbose=2)
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
    with ft.Device(ft.GPU(), 0):

        n = 4

        # Add verbose=1 to see the resulting native code
        @ft.optimize(
            # Parallel Loop Li as GPU threads
            schedule_callback=lambda s: s.parallelize('Li', 'threadIdx.x'))
        def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
            y = ft.empty((n,), "int32")
            #! nid: Li # Name the loop below as "Li"
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
    with ft.Device(ft.GPU(), 0):

        @ft.optimize(
            # Parallel Loop Li as GPU threads
            schedule_callback=lambda s: s.parallelize("Li", "threadIdx.x"))
        # Use "byvalue" for `n` so it can be used both during kernel launching
        # and inside a kernel
        def test(n: ft.Var[(), "int32", "input", "byvalue"], a, b):
            a: ft.Var[(n,), "int32"]
            b: ft.Var[(n,), "int32"]
            y = ft.empty((n,), "int32")
            #! nid: Li # Name the loop below as "Li"
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
        #! nid: Li  # <-- 1. Name the loop as Li
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
        #! nid: Li
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
def test_vector_add_pytorch():
    # Used in docs/index.md and docs/guide/schedules.md

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
