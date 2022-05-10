# FreeTensor

**For PLDI '22 Artifact Evaluation, please follow the README uploaded to the submission site, where you can find step-by-step instructions.**

Write and optimize high-performance native loop-based tensor programs in Python.

## Features

Write a simple vector addition with loops that compiles to native code:

```python
import freetensor as ft
import numpy as np

n = 4

# Change this line to ft.optimize(verbose=1) to see the resulting native code
@ft.optimize
def test(a: ft.Var[(n,), "int32"], b: ft.Var[(4,), "int32"]):
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
        'nid: Li'  # Name the loop below as "Li"
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array(4, dtype="int32"),
             np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)
```


## Dependencies

- Linux
- Python (>= 3.8, for the Python frontend)
- GCC (>= 8, to support C++17 and the "unroll" pragma)
- CUDA (>= 10.2, to support GCC 8, Optional)
- MKL (Optional)
- Java (Build-time dependency only)

## Build

First, clone this repo. Don't forget there are some submodules.

```sh
git clone --recursive <path-to-this-repo>
```

Then, build.

```sh
mkdir build
cd build
cmake ..
make -j
```

If not using CUDA, add a `-DFT_WITH_CUDA=OFF` to `cmake`. If using MKL, add a `-DFT_WITH_MKL=<path/to/mkl/root>` to `cmake`.

It will build a shared library with a name like `freetensor_ffi.cpython-37m-x86_64-linux-gnu.so`.

There are some debugging options. Adding `-DFT_DEBUG_LOG_NODE=ON` to `cmake` enables tracing to tell by which pass a specific AST node is modified. Adding `-DFT_DEBUG_PROFILE` to `cmake` profiles some heavy functions in the compiler.

## Run

To run any program with FreeTensor, one should add the `python/` and `build/` directory to `PYTHONPATH` first.

E.g. to run a python program `a.py` with FreeTensor in the `build/` directory,

```sh
PYTHONPATH=../python:../build:$PYTHONPATH python3 a.py
```

To run the test, first change into the `test/` directory, then

```sh
PYTHONPATH=../python:../build:$PYTHONPATH pytest
```

To run a single test case, specify the test case name, and optionally use `pytest -s` to display the standard output. E.g,

```sh
PYTHONPATH=../python:../build:$PYTHONPATH pytest -s 0.hello_world/test_basic.py::test_hello_world
```

If using GDB, one should invoke PyTest with `python3 -m`:

```sh
PYTHONPATH=../python:../build:$PYTHONPATH gdb --args python3 -m pytest
```

If using Valgrind, one should set Python to use the system malloc:

```sh
PYTHONPATH=../python:../build:$PYTHONPATH PYTHONMALLOC=malloc valgrind python3 -m pytest
```

Sometimes Valgrind is not enough to detect some errors. An alternative is to use the sanitizer from GCC. To use it, first edit `CMakeLists.txt` to add a `-fsanitize=address` compiler flag (or other mode like `-fsanitize=undefined`), then:

```sh
PYTHONPATH=../python:../build:$PYTHONPATH LD_PRELOAD=`gcc -print-file-name=libasan.so` pytest -s
```

## Contribute

Please configure (or install some plugins for) your editor, to support `clang-format`, `yapf` and `editorconfig`, for code formating.

## Code Structure

```
include/ --------------------------------------------------- C++ headers
|- ref.h --------------------------------------------------- A smart pointer, based on std::shared_ptr, used all around the code
|- ast.h --------------------------------------------------- Base class for AST (IR of FreeTensor) nodes
|- stmt.h -------------------------------------------------- Statement nodes of an AST
|- expr.h -------------------------------------------------- Expression nodes of an AST
|- visitor.h ----------------------------------------------- Inherit Visitor in this file to examine an AST
|- mutator.h ----------------------------------------------- Inherit Mutator in this file to modify an AST
|- ffi.h --------------------------------------------------- Interface between C++ and Python. Implementations are in src/ffi/
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
test/ ------------------------------------------------------ Unit tests
```
