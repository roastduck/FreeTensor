# Build and Run

[TOC]

## Dependencies

- Linux
- Python (>= 3.8, for the Python frontend)
- C++ compiler (GCC >= 11 or Clang >= 16, to have enough C++20 support and the "unroll" pragma)
- CUDA (>= 11.4.1, to support GCC 11, Optional, only supported with GCC)
- MKL (Optional)
- PyTorch (Optional, see below)
- Java (= 11, Build-time dependency only)

Other Python dependencies can be installed automatically when installing FreeTensor.

!!! note "Note on Python version"
    Because we are analyzing Python AST, which is sensitive to Python version, there may be potential bugs for Python strictly later than 3.8. Please file an issue if something goes wrong

!!! warning "Conflict with PyTorch"
    FreeTensor can be used together with PyTorch, and FreeTensor provides an optional integration with PyTorch. *With or without* this integration, FreeTensor may conflict with PyTorch if they both linked against some common dependencies but these dependencies are of different versions. This conflict can lead to weird error both at compile time and run time, and silent performance drop. Thus we highly recommand to build FreeTensor and PyTorch locally in the same environment. We also provide scripts to build a docker container for this purpose (see below).

## Build

First, clone this repo. Don't forget there are some submodules.

```sh
git clone --recursive <path/to/this/repo>
```

Then, build and install.

```sh
pip3 install .
```

This command will build FreeTensor with minimal dependencies. To build with a larger feature set, append one or more following `-C--local=???.toml` options to the command. Please note that these options requires a new enough `pip`.

- `pip3 install . -C--local=with-cuda.toml`: Build with CUDA.
- `pip3 install . -C--local=with-mkl.toml`: Build with MKL.
- `pip3 install . -C--local=with-pytorch.toml`: Build with PyTorch.

!!! note "Note if building with PyTorch"
    Since there are conflicts with PyTorch as described above, we do not manage PyTorch as a dependency in the Python project, and it should be installed manually before installing FreeTensor. However, this breaks the requirement of `pip` that all dependencies should be declared, so `pip` must be called with `--no-build-isolation`, and this further requires installing the following build-time dependencies manally: `pip3 install py-build-cmake~=0.1.8 z3-solver setuptools`.

The `.toml` files in these options can be found in the root directory of this repository, in which options to CMake are set. The full set of CMake options of FreeTensor are:

- `-DFT_WITH_CUDA=ON/OFF`: build with/without CUDA (defaults to `ON`).
- `-DFT_WITH_MKL=ON/<path/to/mkl/root>/OFF`: build with MKL (defaults to `OFF`).

    The path accepts by CMake should be a raw unescaped path; i.e. `-DFT_WITH_MKL="/some path"` is good since the quotes are resolved by the shell but `-DFT_WITH_MKL=\"/some\ path\"` is not.

- `-DFT_WITH_PYTORCH=ON/OFF`: build with/without PyTorch integration (including copy-free interface from/to PyTorch), requring PyTorch installed on the system (defaults to `OFF`).
- `-DFT_COMPILER_PORTABLE=ON/OFF`: do not build FreeTensor itself and its dependencies with non-portable instructions (defaults to `OFF`).
- `-DFT_WITH_CCACHE=ON/OFF/AUTO`: use ccache to speed up compilation, which is useful for compiling behind py-build-cmake (defaults to `AUTO`).
- `-DFT_DEBUG_BLAME_AST=ON` (for developers): enables tracing to tell by which pass a specific AST node is modified.
- `-DFT_DEBUG_PROFILE=ON` (for developers): profiles some heavy functions in the compiler.
- `-DFT_DEBUG_SANITIZE=<sanitizer_name>` (for developers): build with GCC sanitizer (set it to a sanitizer name to use, e.g. address).

You can create your own `.toml` files to set them explicitly.

Alternatively, you can build our docker images by `make -f docker.Makefile <variant>`, where `<variant>` can be:

- `minimal-dev`, for `-DFT_WITH_CUDA=OFF -DFT_WITH_MKL=OFF -DFT_WITH_PYTORCH=OFF`, or
- `cuda-mkl-dev`, for `-DFT_WITH_CUDA=ON -DFT_WITH_MKL=ON -DFT_WITH_PYTORCH=OFF`, or
- `cuda-mkl-pytorch-dev`, for `-DFT_WITH_CUDA=ON -DFT_WITH_MKL=ON -DFT_WITH_PYTORCH=ON`.

After installation, simply `import freetensor` to Python to use.

## Global Configurations

There are serveral global configurations can be set via environment variables:

- `FT_PRETTY_PRINT=ON/OFF`. Enable/disable colored printing. If omitted, FreeTensor will guess this option with runtime information.
- `FT_PRINT_ALL_ID=ON/OFF`. Print (or not) IDs of all statements in an AST. Defaults to `OFF`.
- `FT_PRINT_SOURCE_LOCATION=ON/OFF`. Print (or not) Python source location of all statements in an AST. Defaults to `OFF`.
- `FT_FAST_MATH=ON/OFF`. Run (or not) `pass/float_simplify` optimization pass, and enable (or not) fast math on backend compilers. Defaults to `ON`.
- `FT_WERROR=ON/OFF`. Treat warnings as errors (or not). Defaults to `OFF`.
- `FT_BACKEND_COMPILER_CXX=<path/to/compiler>`. The C++ compiler used to compile the optimized program. Default to the same compiler found when building FreeTensor itself, and compilers found in the `PATH` enviroment variable. This environment variable should be set to a colon-separated list of paths, in which the paths are searched from left to right.
- `FT_BACKEND_COMPILER_NVCC=<path/to/compiler>`. The CUDA compiler used to compile the optimized program (if built with CUDA). Default to the same compiler found when building FreeTensor itself, and compilers found in the `PATH` enviroment variable. This environment variable should be set to a colon-separated list of paths, in which the paths are searched from left to right.
- `FT_BACKEND_OPENMP`. Path to an OpenMP library linked to the optimized program. Default to the same library linked to FreeTensor itself. This environment variable should be set to a colon-separated list of paths, in which the libraries are linked from left to right.
- `FT_DEBUG_RUNTIME_CHECK=ON/OFF`. If `ON`, check out-of-bound access and integer overflow at the generated code at runtime. This option is only for debugging, and will introduce significant runtime overhead. Currently the checker cannot print the error site, please also enable `FT_DEBUG_BINARY` and then use GDB to locate the error site (by setting a breakpoint on `exit`). Defaults to `OFF`.
- `FT_DEBUG_BINARY=ON/OFF` (for developers). If `ON`, compile with `-g` at backend. FreeTensor will not delete the binary file after loading it. Defaults to `OFF`.
- `FT_DEBUG_CUDA_WITH_UM=ON/OFF`. If `ON`, allocate CUDA buffers on Unified Memory, for faster (debugging) access of GPU `Array` from CPU, but with slower `Array` allocations and more synchronizations. No performance effect on normal in-kernel computations. Defaults to `OFF`.

This configurations can also set at runtime in [`ft.config`](../../api/#freetensor.core.config).

## Run the Tests

To run the test, first change into the `test/` directory, then

```sh
pytest
```

To run a single test case, specify the test case name, and optionally use `pytest -s` to display the standard output. E.g,

```sh
pytest -s 00.hello_world/test_basic.py::test_hello_world
```

!!! note "Debugging (for developers)"

    If using GDB, one should invoke PyTest with `python3 -m`:

    ```
    gdb --args python3 -m pytest
    ```

    If using Valgrind, one should set Python to use the system malloc:

    ```
    PYTHONMALLOC=malloc valgrind python3 -m pytest
    ```

    Sometimes Valgrind is not enough to detect some errors. An alternative is to use the sanitizer from GCC. For example, if you are using the "address" sanitizer, first set `-DFT_DEBUG_SANITIZE=address` to `cmake`, and then:

    ```
    LD_PRELOAD=`gcc -print-file-name=libasan.so` pytest -s
    ```

    If you are using another sanitizer, change the string set to `FT_DEBUG_SANITIZE` and the library's name. For example, `-DFT_DEBUG_SANITIZE=undefined` and `libubsan.so`.

## Build this Document

First install some dependencies:

```sh
pip3 install --user mkdocs mkdocstrings==0.18.1 "pytkdocs[numpy-style]"
```

From the root directory of FreeTensor, run a HTTP server to serve the document (recommended, but without document on C++ interface due to [a limitation](https://github.com/mkdocs/mkdocs/issues/1901)):

```sh
mkdocs serve
```

Or build and save the pages (with document on C++ interface, requiring Doxygen and Graphviz):

```sh
doxygen Doxyfile && mkdocs build
```

!!! note "Publish the documents to GitHub Pages (for developers)"

    `doxygen Doxyfile && mkdocs gh-deploy`
