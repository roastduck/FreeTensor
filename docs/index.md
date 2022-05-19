# FreeTensor

A language and compiler for irregular tensor programs.

- [GitHub](https://github.com/roastduck/FreeTensor)
- [User Guide](guide)
- [API Reference](api)
- [Publication](about/pub)
- [License](https://github.com/roastduck/FreeTensor/blob/master/LICENSE)

## Features by Example

Write a simple vector addition with loops that compiles to native code:

```python
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
```

If you are not willing to compile the program once for each different `n`, you can set `n` as another function argument (but you may lose some performance). In FreeTensor, all variables are tensors, where scalars are 0-D tensors.

```python
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
```

If building with CUDA, you can also run the program on a GPU. This time, a "[schedule](guide/schedules)" (an explicit program transformation) is needed, and memory types of variables should be properly set.

```python
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
```

Some common tensor operations, including tensor addition (broadcasting is supported), are pre-defined functions in FreeTensor. They are defiend in [`freetensor.libop`](api/#freetensor.libop), and they can also be invoked using operator overloading. These functions are pure Python functions, which will be inlined into your code, and will enjoy a joint optimization.

```python
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
```

FreeTensor also supports reverse-mode Automatic Differentiation:

```python
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
```

