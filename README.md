# IR

## Build

```sh
mkdir build
cd build
cmake ..
make -j
```

It will build a shared library with a name like `ffi.cpython-37m-x86_64-linux-gnu.so`.

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

To run a single test case, use `pytest -s` to specify the test case name. E.g,

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

## Contribute

Please configure (or install some plugins for) your editor, to support `clang-format` and `editorconfig`, for code formating.
