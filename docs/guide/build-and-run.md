# Build and Run

## Dependencies

- Linux
- Python (>= 3.8, for the Python frontend)
- GCC (>= 8, to support C++17 and the "unroll" pragma)
- CUDA (>= 10.2, to support GCC 8, Optional)
- MKL (Optional)
- Java (Build-time dependency only)

Python dependencies:

```sh
pip3 install --user numpy sourceinspect astor Pygments xgboost
```

!!! note "Note on Python version"
    Because we are analyzing Python AST, which is sensitive to Python version, there may be potential bugs for Python strictly later than 3.8. Please file an issue if something goes wrong

!!! note "Note on future changes"
    We have a plan to migrade to C++20 in a near future, which requires GCC >= 10

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

## Global Configurations

There are serveral global configurations can be set via environment variables:

- `FT_PRETTY_PRINT=ON/OFF`. Enable/disable colored printing.
- `FT_PRINT_ALL_ID=ON/OFF`. Print (or not) IDs of all statements in an AST.
- `FT_WERROR=ON/OFF`. Treat warnings as errors (or not).

This configurations can also set at runtime in [`ft.config`](../../api/#freetensor.core.config).

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

## Build this Document

First install some dependencies:

```sh
pip3 install --user mkdocs mkdocstrings "pytkdocs[numpy-style]"
```

From the root directory of FreeTensor, run a HTTP server to serve the document (recommended):

```sh
PYTHONPATH=./python:./build:$PYTHONPATH mkdocs serve
```

Or build and save the pages:

```sh
PYTHONPATH=./python:./build:$PYTHONPATH mkdocs build
```

!!! note "Publish the documents to GitHub Pages (for developers)"

    `PYTHONPATH=./python:./build:$PYTHONPATH mkdocs gh-deploy`
