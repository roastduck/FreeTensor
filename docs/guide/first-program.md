# Your First Program with FreeTenor

In this page, we introduce some basic concepts of FreeTensor.

## Example: Vector addition

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

Here is a basic example program in FreeTensor. You write a Python function that manipulates FreeTensor's tensor type `ft.Var`, decorate the function with `ft.optimize`, and finally invoke the decorated function. FreeTensor will generate C++ code for this vector addition, compile it using a native compiler, and finally load it back to Python. Set `verbose = 1` to `optimize` if you are interested in the generated native code.

To write such a function, you need to follow some basic concept described in this page.

# Declare and Define Tensors

All tensors, including function parameters, intermediate tensors and return values should be properly declared or defined.

Function parameters should be declared like [`x : ft.Var[shape, data_type]`](/api/#freetensor.core.transformer.Var). Declaring a parameter either in the function signature or as a stand-alone statment is acceptable. If your parameter uses another parameter as shape, you will need the latter manner. An optional parameter `atype` can be set to `"output"` or `"inout"` if you want to mutate a function argument.

Intermediate and returning tensors can be created by [`ft.empty`](/api/#freetensor.core.transformer.empty), [`ft.var`](/api/#freetensor.core.transformer.var) or [`ft.zeros`](/api/#freetensor.libop.constant.zeros). If you are using FreeTensor for GPU computing, an optional parameter `mtype` can be set to specify where to store the tensor. It defaults to the main memory of your currently chosen computing device.

All tensors and their slices are implemented by an internal [`ft.VarRef`](/api/#freetensor.core.expr.VarRef) type. If you are looking for a tensor's API, `ft.VarRef` is the right place.

# Dynamic or Static

Another concept is that statements and expressions in your program are divided into two categories: *dynamic* and *static*. Dynamic statements or expressions are restricted to a small subset of Python, and are *compiled* to native code. Static statements or expressions can be any Python statements or expressions, and are executed *before* compilation. In other words, static statements or expressions are like macros or templates in C++, while dynamic ones are actually quotations in [Multi-Stage Programming](https://en.wikipedia.org/wiki/Multi-stage_programming).

The following statements and expressions are considered dynamic:

- Declarations, definitions and operations of FreeTensor's tensor type `ft.Var` (or its internal implementation `ft.VarRef`).
- `if` statements, `for ... in range(...)` and `assert` statements that have a `ft.Var` condition or range.

All other statements and expressions are considered static.

With the help of dynamic and static categories, you can utilize complex Python functions as the static part, while still generate high-performance native code using dynamic loops. For example, the following code combines static and dynamic code to sum multiple vectors together:

```python
import freetensor as ft
import numpy as np

n = 4

@ft.optimize
def test(a: ft.Var[(n,), "int32"], b: ft.Var[(4,), "int32"],
         c: ft.Var[(4,), "int32"]):
    inputs = [a, b, c]  # Static
    y = ft.empty((n,), "int32")  # Dynamic
    for i in range(n):  # Dyanmic
        y[i] = 0  # Dynamic
        for item in inputs:  # Static
            y[i] += item[i]  # Dynamic
    return y

y = test(np.array([1, 2, 3, 4], dtype="int32"),
         np.array([2, 3, 4, 5], dtype="int32"),
         np.array([3, 4, 5, 6], dtype="int32")).numpy()
print(y)
```

However, there might be some counterintuitive behaviours when using static statments or expressions. Please remember that static static statements or expressions are executed *before* compilation, so the following piece of code will result in a list containing only one item: the expression `i`, instead of 10 numbers:

```python
lst = []
for i in range(10):  # Dynamic
    lst.append(i)  # Static. Appends only once
```
