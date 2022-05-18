# FreeTensor

Write and optimize high-performance native loop-based tensor programs in Python.

## Features

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

If building with CUDA, you can also run the program on a GPU. This time, a "schedule" (an explicit program transformation) is needed, and memory types of variables should be properly set.

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

Some common tensor operations, including tensor addition (broadcasting is supported), are pre-defined functions in FreeTensor. They are defiend in `freetensor.libop`, and they can also be invoked using operator overloading. These functions are pure Python functions, which will be inlined into your code, and will enjoy a joint optimization.

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

## Get Started

[Get Started](https://roastduck.github.io/FreeTensor/guide/)

## Contribute

Please configure (or install some plugins for) your editor, to support `clang-format`, `yapf` and `editorconfig`, for code formating.

## Code Structure

```
ffi/ ------------------------------------------------------- Interface between C++ and Python
grammar/ --------------------------------------------------- ANTLR grammar files used for serialization
include/ --------------------------------------------------- C++ headers
|- ref.h --------------------------------------------------- A smart pointer, based on std::shared_ptr, used all around the code
|- ast.h --------------------------------------------------- Base class for AST (IR of FreeTensor) nodes
|- stmt.h -------------------------------------------------- Statement nodes of an AST
|- expr.h -------------------------------------------------- Expression nodes of an AST
|- visitor.h ----------------------------------------------- Inherit Visitor in this file to examine an AST
|- mutator.h ----------------------------------------------- Inherit Mutator in this file to modify an AST
|- schedule.h ---------------------------------------------- All user specified transformations (schedules). Main interface. Details are in schedule/
|- frontend/ ----------------------------------------------- C++ utilities used in Python API
|- math/ --------------------------------------------------- Math utilities
|- schedule/ ----------------------------------------------- All user specified transformations (schedules)
|- pass/ --------------------------------------------------- All user agnostic transformations (used inside or after schedules)
|- analyze/ ------------------------------------------------ Passes to extract information from an AST
|- codegen/ ------------------------------------------------ Passes to generate a target code from an AST
`- driver/ ------------------------------------------------- Infrastructure to run a generated target code
src/ ------------------------------------------------------- C++ sources (the structure is almost same with include/)
python/ ---------------------------------------------------- Python API
runtime/ --------------------------------------------------- (Minimal) runtime code to be compiled into target exexutables
test/ ------------------------------------------------------ Unit tests
```

## Publication

Shizhi Tang, Jidong Zhai, Haojie Wang, Lin Jiang, Liyan Zheng, Zhenhao Yuan, and Chen Zhang. 2022. FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs. In *Proceedings of the 43rd ACM SIGPLAN International Conference on Programming Language Design and Implementation (PLDI ’22), June 13-17, 2022, San Diego, CA, USA*. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3519939.3523448. ([Download](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/pldi22-freetensor/))

```
@inproceedings{freetensor,
  author = {Tang, Shizhi and Zhai, Jidong and Wang, Haojie and Jiang, Lin and Zheng, Liyan and Yuan, Zhenhao and Zhang, Chen},
  title = {FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs},
  year = {2022},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3519939.3523448},
  doi = {10.1145/3519939.3523448},
  booktitle = {Proceedings of the 43rd ACM SIGPLAN International Conference
  on Programming Language Design and Implementation (PLDI ’22)},
  numpages = {16},
  location = {San Diego, CA, USA},
  series = {PLDI ’22}
}
```

**NOTE: API of FreeTensor has been changed since submission. To reproduce the exact result in the paper, please consider the Artifact Evaluation version of FreeTensor, published [here](https://zenodo.org/record/6327595).**
