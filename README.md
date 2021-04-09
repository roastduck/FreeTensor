# IR

## Dependencies

- Python (>= 3.8, for the Python frontend)
- GCC (>= 8, to support C++17 and the "unroll" pragma)
- CUDA (>= 10.2, to support GCC 8)

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

It will build a shared library with a name like `ffi.cpython-37m-x86_64-linux-gnu.so`.

You can also replace `cmake ..` with `cmake -DIR_DEBUG=ON ..` to enable some debugging features. Currently, this includes:

- Enable tracing to tell by which pass a specific AST node is modified.

## Run

To run any program with IR, one should add the `python/` and `build/` directory to `PYTHONPATH` first.

E.g. to run a python program `a.py` with IR in the `build/` directory,

```sh
PYTHONPATH=../python:../build:$PYTHONPATH python3 a.py
```

To run the test, first change into the `test/` directory, then

```sh
PYTHONPATH=../python:../build:$PYTHONPATH pytest
```

To run a single test case, specify the test case name, and optionally use `pytest -s` to display the standard output. E.g,

```sh
PYTHONPATH=../python:../build:$PYTHONPATH pytest -s codegen/test_basic.py::test_hello_world
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

Please configure (or install some plugins for) your editor, to support `clang-format` and `editorconfig`, for code formating.

## Code Structure

```
include/ --------------------------------------------------- C++ headers
|- ref.h --------------------------------------------------- A smart pointer, based on std::shared_ptr, used all around the code
|- ast.h --------------------------------------------------- Base class for AST (the form of our IR) nodes
|- stmt.h -------------------------------------------------- Statement nodes of an AST
|- expr.h -------------------------------------------------- Expression nodes of an AST
|- visitor.h ----------------------------------------------- Inherit Visitor in this file to examine an AST
|- mutator.h ----------------------------------------------- Inherit Mutator in this file to modify an AST
|- ffi.h --------------------------------------------------- Interface between C++ and Python. Implementations are in src/ffi/
|- schedule.h ---------------------------------------------- All user specified transformations (schedules). Main interface. Details are in schedule/
|- schedule/ ----------------------------------------------- All user specified transformations (schedules)
|- pass/ --------------------------------------------------- All user agnostic transformations (used inside or after schedules)
|- analyze/ ------------------------------------------------ Passes to extract information from an AST
|- codegen/ ------------------------------------------------ Passes to generate a target code from an AST
`- driver/ ------------------------------------------------- Infrastructure to run a generated target code
src/ ------------------------------------------------------- C++ sources (the structure is almost same with include/)
python/ ---------------------------------------------------- Python API
test/ ------------------------------------------------------ Unit tests
```
