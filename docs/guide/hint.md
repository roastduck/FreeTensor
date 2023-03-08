# Optimize a Program with Hints

[TOC]

Expressions and statements in a program do not always provide enough mathematical information for the compiler. Since the compiler must ensure safety for all possible cases, optimizations might be missed. In FreeTensor, you may provide additional information in some ways to guide the compiler, in other words, guide FreeTensor.

## Types with Sign Information

Suppose you are filling a `n` by `m` matrix from `0` to `n * m - 1` in row-major order, you may loop from `0` to `n * m - 1` with a single loop, and the iterator `i` by `m` to find each element in the `i // m`-th row and `i % m`-th column:

```python
y = ft.empty((n, m), "int32")
for i in range(n * m):
    y[i // m, i % m] = i
```

Definitely there are other solutions, for example using two loops, but we are going to use the single-loop program to show a common but unobvious performance pitfall: Integer division in Python (including in FreeTensor) rounds to negative infinity, but integer division in most target instructions and target languages like C++ or CUDA rounds to 0. There is only a difference when dividend is negative, but compiling a general Python division to target architectures has to involve an extra branch to check it. In our example, `n` and `m` refer to the shape of a matrix, so it cannot be negative. If we can hint FreeTensor, we can avoid the redundant branch.

FreeTensor supports adding a suffix to the data type string to show the sign of a number. Simply changing `"int32"` to `"int32>=0"` will make a difference. All supported suffices are `">0"`, `>=0`, `<0`, `<=0`, `!=0` and `==0`. A complete example is below:

```python
import freetensor as ft

print("Without hint")

@ft.optimize(verbose=1)  # `verbose=1` prints the code
def test_no_hint(n: ft.Var[(), "int32"], m: ft.Var[(), "int32"]):
    y = ft.empty((n, m), "int32")
    for i in range(n * m):
        y[i // m, i % m] = i
    return y

# You will find `runtime_mod` in the code, which involves additional branching
assert "runtime_mod" in test_no_hint.native_code()
assert "%" not in test_no_hint.native_code()

print("With hint")

@ft.optimize(verbose=1)  # `verbose=1` prints the code
def test_hint(n: ft.Var[(), "int32"], m: ft.Var[(), "int32>=0"]):
    y = ft.empty((n, m), "int32")
    for i in range(n * m):
        y[i // m, i % m] = i
    return y

# You will find native C++ `%` in the code, which compiles directly to mod
# instructions
assert "runtime_mod" not in test_hint.native_code()
assert "%" in test_hint.native_code()
```

The sign hint also works for other optimizations. One example is `ft.sqrt(x * x)` can be automatically optimized to `x` if `x` is non-negative. Another example is `ft.min(a, b)` can be automatically optimized to `a` if `a` is negative while `b` is positive.

## Provide Information by Assertions

Another way to hint FreeTensor is to add some `assert` statements in the program. In this way, you can add some more precise hints, which reveals mathematical properties among specifc elements, instead of the whole tensor. Here is an example of adding two `n`-length vectors, and the program is [scheduled](../schedules) to execute in parallel.

```python
def sch(s):
    outer, inner = s.split('Li', 32)
    s.parallelize(outer, 'openmp')

@ft.optimize(schedule_callback=sch, verbose=1)
def test(n: ft.Var[(), "int32"], a, b):
    a: ft.Var[(n,), "int32"]
    b: ft.Var[(n,), "int32"]
    y = ft.empty((n,), "int32")
    #! label: Li
    for i in range(n):
        y[i] = a[i] + b[i]
    return y
```

The algorithm is simple: we use (at most) `n // 32` threads, each computing 32 elements. However, if you look the code, you will find the length of the serial loop is not as simple as 32. Instead, it is a complex expression that results in 31 or 32. This is because `n` is not always divisible by `32`.

Suppose in our case, `n` is really divisible by 32, we can add an `assert` statement to hint FreeTensor: `assert n % 32 == 0`, and the serial loop will have a neat length 32. A complete example is below:

```python
import freetensor as ft
import re

def sch(s):
    outer, inner = s.split('Li', 32)
    s.parallelize(outer, 'openmp')

@ft.optimize(schedule_callback=sch, verbose=1)
def test_no_hint(n: ft.Var[(), "int32"], a, b):
    a: ft.Var[(n,), "int32"]
    b: ft.Var[(n,), "int32"]
    y = ft.empty((n,), "int32")
    #! label: Li
    for i in range(n):
        y[i] = a[i] + b[i]
    return y

# You will not find a 32-length loop
assert not re.search(r".* = 0; .* < 32; .*\+\+", test_no_hint.native_code())

@ft.optimize(schedule_callback=sch, verbose=1)
def test_hint(n: ft.Var[(), "int32"], a, b):
    a: ft.Var[(n,), "int32"]
    b: ft.Var[(n,), "int32"]
    y = ft.empty((n,), "int32")
    assert n % 32 == 0
    #! label: Li
    for i in range(n):
        y[i] = a[i] + b[i]
    return y

# You will find a 32-length loop
assert re.search(r".* = 0; .* < 32; .*\+\+", test_hint.native_code())
```
