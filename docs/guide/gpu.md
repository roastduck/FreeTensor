# Running on a GPU

## Example: Vector addition on a GPU

If FreeTensor is built with a CUDA backend, you can compile your program to a GPU. We still take a vector addition as an example:

```python
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
```

Similar to [parallelizing to OpenMP threads](../schedules/#example-parallel-vector-addition), in this example, we parallelize Loop `Li` to the `threadIdx.x` dimension of CUDA. There are two major differences:

1. You are now calling `parallelize` schedule with a `threadIdx.x` parameter, instead of `openmp`.
2. All the code are enclosed by a `with ft.GPU(0)` scope.

Usually, you not only parallelize your loops to `threadIdx.x`, but also other CUDA dimensions like `blockIdx.x`. To achieve this, you either parallelize different loops in a loop nests to different CUDA dimensions, or [`split`](../../api/#freetensor.core.schedule.Schedule.split) your loops before parallelizing them.

As for the `with ft.GPU(0)` scope, `ft.GPU(0)` specifies a [`Device`](../../api/#freetensor.core.driver.Device) (a specific hardware device of GPU). By calling `with` on a device, default values of several classes and functions are set, but currently you only need to be aware of two things:

1. It sets the `Device` of `optimize`.
2. It sets the default `mtype` of all tensors in the program, which is an optional parameter of `ft.Var`, `ft.empty`, etc.

`mtype` refers to memory type. It controls where a tensor is stored. It defaults to `"cpu"` for a CPU program, and `"gpu/global"` for a GPU program. You probably GPU requires putting each variable to a right place (global memory, shared memory, registers, etc.), and this can be done by setting `mtype`s of each tensor. There are several ways to set `mtype`s:

1. (Recommended) Leave them to the default `"gpu/global"` first, and modify them with the [`set_mem_type`](../../api/#freetensor.core.schedule.Schedule.set_mem_type) schedule. In this way, you write some architecture-dependent schedules, but keep your function architecture-independent.
2. (Experimental) Leave them to the default `"gpu/global"` first, and modify them automatically using [`auto_schedule`](../../api/#freetensor.core.schedule.Schedule.auto_schedule), or the [`auto_set_mem_type`](../../api/#freetensor.core.schedule.Schedule.auto_set_mem_type) schedule (which is a part of `auto_schedule`).
3. Set them explicitly in the program by setting an optional `mtype` parameter of `ft.Var`, `ft.empty`, etc.

## `mtype="byvalue"` for Dynamic Tensor Shapes

Tensors with normal `mtypes` (`"cpu"`, `"gpu/global"`, etc.) are passed by references, which means a `"cpu"` tensor can only be accessed from a CPU, and a `"gpu/global"` tensor can only be accessed from a GPU. However, sometimes, and especially for dynamic tensor shapes, we want the shapes to be passed by values, and accessible from both CPUs and GPUs (remember we need tensor's shape both when launching a kernel from the CPU side, and during actual computatoin on the GPU side). In this case, we can set the shape-related tensors a `"byvalue"` `mtype`, and here is an example:

```python
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
```
