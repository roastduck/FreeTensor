# Build and Run

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
make -j  # Or use Ninja
```

There are some options to `cmake`:

- `-DFT_WITH_CUDA=ON/OFF`: build with/without CUDA (defaults to `ON`).
- `-DFT_WITH_MKL=<path/to/mkl/root>`: build with MKL (path to MKL is required, defaults to building without it).
- `-DFT_DEBUG_LOG_NODE=ON` (for developers): enables tracing to tell by which pass a specific AST node is modified.
- `-DFT_DEBUG_PROFILE` (for developers): profiles some heavy functions in the compiler.

It will build a shared library with a name like `freetensor_ffi.cpython-37m-x86_64-linux-gnu.so`, which can be used in Python via `import freetensor`.

## Run a Program with FreeTensor

To run any program with FreeTensor, one should add the `python/` and `build/` directory to `PYTHONPATH` first.

E.g. to run a python program `a.py` with FreeTensor in the `build/` directory,

```sh
PYTHONPATH=../python:../build:$PYTHONPATH python3 a.py
```

## Run the Tests

To run the test, first change into the `test/` directory, then

```sh
PYTHONPATH=../python:../build:$PYTHONPATH pytest
```

To run a single test case, specify the test case name, and optionally use `pytest -s` to display the standard output. E.g,

```sh
PYTHONPATH=../python:../build:$PYTHONPATH pytest -s 0.hello_world/test_basic.py::test_hello_world
```

!!! note "Debugging (for developers)"

    If using GDB, one should invoke PyTest with `python3 -m`:

    ```
    PYTHONPATH=../python:../build:$PYTHONPATH gdb --args python3 -m pytest
    ```

    If using Valgrind, one should set Python to use the system malloc:

    ```
    PYTHONPATH=../python:../build:$PYTHONPATH PYTHONMALLOC=malloc valgrind python3 -m pytest
    ```

    Sometimes Valgrind is not enough to detect some errors. An alternative is to use the sanitizer from GCC. To use it, first edit `CMakeLists.txt` to add a `-fsanitize=address` compiler flag (or other mode like `-fsanitize=undefined`), then:

    ```
    PYTHONPATH=../python:../build:$PYTHONPATH LD_PRELOAD=`gcc -print-file-name=libasan.so` pytest -s
    ```

