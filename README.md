# FreeTensor

[Get Started](https://roastduck.github.io/FreeTensor/guide/) | [Contributing](https://roastduck.github.io/FreeTensor/about/contrib/) | [Publication](https://roastduck.github.io/FreeTensor/about/pub/) | [License](https://github.com/roastduck/FreeTensor/blob/master/LICENSE)

Write and optimize high-performance native loop-based tensor programs in Python.

## Features

- Compiling loop-based tensor programs in Python to native code
- Dynamic tensor shapes supported
- Accept explicit transformations to programs for optimization, including parallelization, loop transformations and memory optimizations
- Parallelization with OpenMP or CUDA
- Reverse-mode Automatic Differentiation

[Features by Example](https://roastduck.github.io/FreeTensor/#features-by-example)

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

## Acknowledgement

Many designs in FreeTensor are inspired by [TVM](https://github.com/apache/tvm/). Thank the TVM community for their contributions on tensor program compiling.
