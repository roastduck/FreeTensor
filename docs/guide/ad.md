# Automatic Differentiation

[TOC]

Automatic Differentiation (AD) transforms a program to another program that computes the original one's derivative or gradient. FreeTensor supports Reverse-Mode AD, and there is a plan to support Forward-Mode AD in the future.

## Reverse-Mode AD

Suppose there is a program `x -> y -> z -> w` that computes an output `w` from intermediate variables `z` and `y`, and an input variable `x`. Reverse-Mode AD generates a gradient program `dw/dw=1 -> dw/dz -> dw/dy -> dw/dx` that computes `dw/dx` by Chain Rule. `y`, `z` and `w` may be saved in a "tape" when evaluation the original program, to be reused in the gradient one.

If FreeTensor is built with `WITH_PYTORCH=ON`, you can skip this section and turn to the [`@optimize_to_pytorch` integration](../first-program/#copy-free-interface-fromto-pytorch), which integrates seamlessly with PyTorch's autograd mechanism, but will incur some runtime overhead.

Here is an example of Reverse-Mode AD in FreeTensor:

```python
import freetensor as ft
import numpy as np

n = 4

@ft.optimize
@ft.grad(requires=['a', 'b'], provides=[ft.Return()], attach_backward=True)
def test(a: ft.Var[(n,), "float32"], b: ft.Var[(n,), "float32"]):
    y = ft.zeros((), "float32")
    for i in range(n):
        y[()] += a[i] * b[i]
    return y

a = np.array([0, 1, 2, 3], dtype="float32")
b = np.array([3, 2, 1, 0], dtype="float32")
y = test(a, b)
print(y.numpy())
dzdy = np.array(1, dtype='float32')
input_grads = test.input_name_to_gradient_name
output_grads = test.output_name_to_gradient_name
dzda, dzdb = test.backward(
    **{output_grads[ft.Return()]: dzdy})[input_grads['a'], input_grads['b']]
print(dzda.numpy())
print(dzdb.numpy())
```

You need to call [`ft.grad`](../../api/#freetensor.core.autograd.grad) (or the inplace version [`ft.grad_`](../../api/#freetensor.core.autograd.grad_)) to generate a *forward* function and a *backward* function. In this example, the backward function is attached as the `test.backward` property because `attach_backward` is set to `True`. You can set it to `False` and `ft.grad` will return both functions. Please note that `test` is updated by `ft.grad` and becomes different than the original function, as it may save some intermediate tensors to a global `tape`, and it must be executed before the backward `test.backward`.

!!! note "Note on JIT"

    [JIT](../first-program/#just-in-time-jit-compilation) is only supported when `attach_backward = True`.

After that, you call `ft.optimize` to optimize and compile the program just as in previous examples. This time it is done for both `test` and `test.backward`.

Finally, you execute `test` and `test.backward`. The parameters and return values of `test.backward` are the gradients of `a`, `b` and `y`, which have their own names. To set and get these parameters and return values, you look up for them in two dictionaries `test.input_name_to_gradient_name` and `test.output_name_to_gradient_name` (in type [`ft.ParamRetDict`](../../api/#freetensor.core.autograd.ParamRetDict). These two dictionaries accept either a name of a parameter, or a special [`ft.Return`](../../api/#freetensor.core.autograd.Return) to specify a return value. When invoking `test.backward`, parameters can be set via keyword arguments, and return values can be collect via a bracket (from a special type [`ft.ReturnValuesPack`](../../api/#freetensor.core.driver.ReturnValuesPack)). These two maps are attached to `test` because `attach_backward` is `True`. Otherwise, they are returned as return values from `ft.grad`.

Intermediate variables are not always have to be saved to the "tape" from the forward function. If a variable is need in the backward function but not saved, it will be re-computed, which is sometimes even faster than saving it due to better locality. By default, FreeTensor uses heuristics to determine which variable to save. To get better performance, you may want to control which intermediate variables should be saved by setting an optional `tapes` parameter in `ft.grad`. `tapes` can either be a different mode, or a explicit list of AST node IDs of all `VarDef` nodes of the variables you want to save.

## Providing Your Custom Gradients

### Why or When do We Need Custom Gradients

Sometimes neither reverse-mode or forward-mode AD produces the most elegant form of gradients. FreeTensor allows you to provide your own gradients for part of the program.

Take softmax as an example: The $\mathbf{y} = softmax(\mathbf{x})$ function is **mathematically defined** by the following steps:

$$\begin{align}
e_i &= \mathrm{e}^{x_i} \label{eq:softmax-1} \\
s &= \sum_i{e_i} \label{eq:softmax-2} \\
y_i &= \frac{e_i}{s} \label{eq:softmax-3}
\end{align}$$

Suppose the final output of the program (the loss) is $z$. If using reverse-mode AD, the gradient of the input: $\frac{\partial z}{\partial x}$ can be computed by the following steps:

$$\begin{align}
\frac{\partial z}{\partial s} &= -\sum_i{\frac{\partial z}{\partial y_i} \frac{y_i}{s}} \label{eq:softmax-grad-1} \\
\frac{\partial z}{\partial e_i} &= \frac{\partial z}{\partial y_i} \frac{1}{s} + \frac{\partial z}{\partial s} \label{eq:softmax-grad-2} \\
\frac{\partial z}{\partial x_i} &= \frac{\partial z}{\partial e_i} e_i \label{eq:softmax-grad-3}
\end{align}$$

However, usually we can NOT compute softmax by Equation $\eqref{eq:softmax-1}\eqref{eq:softmax-2}\eqref{eq:softmax-3}$ for numerical stability issues. Pratically, we **compute** softmax with additional normalization on $\mathbf{x}$:

$$\begin{align}
m &= \max_i{x_i} \label{eq:softmax-norm-1} \\
e_i &= \mathrm{e}^{x_i - m} \label{eq:softmax-norm-2} \\
s &= \sum_i{e_i} \label{eq:softmax-norm-3} \\
y_i &= \frac{e_i}{s} \label{eq:softmax-norm-4}
\end{align}$$

If we directly apply reverse-mode AD on Equation $\eqref{eq:softmax-norm-1}\eqref{eq:softmax-norm-2}\eqref{eq:softmax-norm-3}\eqref{eq:softmax-norm-4}$, the backward program will be like:

$$\begin{align}
\frac{\partial z}{\partial s} &= -\sum_i{\frac{\partial z}{\partial y_i} \frac{y_i}{s}} \\
\frac{\partial z}{\partial e_i} &= \frac{\partial z}{\partial y_i} \frac{1}{s} + \frac{\partial z}{\partial s} \\
\frac{\partial z}{\partial m} &= -\sum_i{\frac{\partial z}{\partial e_i}e_i} \\
\frac{\partial z}{\partial x_i} &= \frac{\partial z}{\partial e_i} e_i + \begin{cases}\frac{\partial z}{\partial m}, &i = \arg\max_j{x_j} \\ 0, &i \neq \arg\max_j{x_j}\end{cases}
\end{align}$$

You may have found that there is an extra $\frac{\partial z}{\partial m}$ involved. Apparently, the gradient should be the same no matter if we do the normalization. This is because $\frac{\partial z}{\partial m}$ actually always equals to $0$. FreeTensor can not dig out this mathematical property, so the computation on $\frac{\partial z}{\partial m}$ will remain and will be wasted.

### How to Write Custom Gradients in FreeTensor

The following examples will demonstrate how to provide your own custom gradients, to override the default AD behaviour. **Please note that this is only for demonstration. If you are just going to use softmax, call it from [`libop.softmax`](../../api/#freetensor.libop.softmax), which has already implemented the following code.**

First we show a softmax implementation with full AD:

```python
import freetensor as ft
import torch

n = 4

@ft.optimize  # Set verbose=1 to see the code
@ft.grad(requires=['x'], provides=[ft.Return()], attach_backward=True)
def test(x: ft.Var[(n,), "float32"]):
    # Automatically decide gradients for this statement
    m = ft.reduce_max(x, axes=[-1])
    e = ft.exp(x - m)
    s = ft.reduce_sum(e, axes=[-1])
    y = e / s
    return y

# Check forward result
x = torch.rand(n, dtype=torch.float32)
x.requires_grad = True
y_ft = test(x).torch()
y_torch = torch.softmax(x, axis=-1)
assert torch.all(torch.isclose(y_ft, y_torch))

# Check backward result
y_torch.grad = dzdy = torch.rand(n, dtype=torch.float32)
input_grads = test.input_name_to_gradient_name
output_grads = test.output_name_to_gradient_name
dzdx_ft = test.backward(**{output_grads[ft.Return()]: dzdy}).torch()
y_torch.backward(y_torch.grad)
dzdx_torch = x.grad
assert torch.all(torch.isclose(dzdx_ft, dzdx_torch, 1e-4, 1e-7))
```

Then, we add our own gradient to it:

```python
import freetensor as ft
import torch

n = 4

@ft.optimize  # Set verbose=1 to see the code
@ft.grad(requires=['x'], provides=[ft.Return()], attach_backward=True)
def test(x: ft.Var[(n,), "float32"]):
    # Mark the range that you want to provide graident for, with `StmtRange`
    with ft.StmtRange() as rng:
        m = ft.reduce_max(x, axes=[-1])
        e = ft.exp(x - m)
        s = ft.reduce_sum(e, axes=[-1])
        y = e / s

        # Call `push_for_backward` so we can use forward values in backward
        e_now = ft.push_for_backward(e)
        s_now = ft.push_for_backward(s)
        y_now = ft.push_for_backward(y)
    # Define gradient in `UserGrad`
    with ft.UserGrad(x, y, stmt_range=rng) as (dzdx, dzdy):
        # Retrieve forward value from `y_now`, NOT `y`
        dzds = -ft.reduce_sum(dzdy * y_now, axes=[-1]) / s_now
        dzde = dzdy / s_now + dzds
        dzdx[...] += dzde * e_now  # Use `+=` here
    return y

# Check forward result
x = torch.rand(n, dtype=torch.float32)
x.requires_grad = True
y_ft = test(x).torch()
y_torch = torch.softmax(x, axis=-1)
assert torch.all(torch.isclose(y_ft, y_torch))

# Check backward result
y_torch.grad = dzdy = torch.rand(n, dtype=torch.float32)
input_grads = test.input_name_to_gradient_name
output_grads = test.output_name_to_gradient_name
dzdx_ft = test.backward(**{output_grads[ft.Return()]: dzdy}).torch()
y_torch.backward(y_torch.grad)
dzdx_torch = x.grad
assert torch.all(torch.isclose(dzdx_ft, dzdx_torch, 1e-4, 1e-7))
```

First, **we mark the range of code that we want to provide gradient for, with `ft.StmtRange`,** as a name `rng`. In the range, we write the code to compute `softmax` as usual. Additionaly, for the values that we want to reuse in the gradient, **we call `ft.push_for_backward` to save it.** `push_for_backward` returns a handle that you can use as a usual tensor in the gradient code. If your `StmtRange` is inside an outer loop, the handle will always reflect the correct iteration (see the next example). Besides, `push_for_backward` does not mean the value will be physically saved in tape: it only means the value will be logically reused in the backward, no matter by saving or by recomputing. `push_for_backward` is orthogonal with the `tapes` parameter in `ft.grad`.

Next, **we define our custom gradient with a `ft.UserGrad` scope.** The scopes receives a special parameter `stmt_range`, which should be set to the `StmtRange` we have just defined. Beside `stmt_range`, `UserGrand` receives an arbitrary number of parameters, in this case, `x` and `y`, and returns the same number of variables, `dzdx` and `dzdy`, so we have the mapping between each variable and its gradient. What we are going to do is update `dzdx` from `dzdy`.

We define our gradient code in the `UserGrad` code of Equation $\eqref{eq:softmax-grad-1}\eqref{eq:softmax-grad-2}\eqref{eq:softmax-grad-3}$. We want to use the forward value `y`, `s` and `e`. But **do NOT directly use its name, use the `push_for_backward` handler `y_now`, `s_now` and `e_now` instead.** Finally, plase note that **we update `dzdx` with `+=` instead of `=`,** because we may be only computing a partial derivative: there may be other functions of `x` other than `y`.

And it is all done.

### Additional Descriptions on `push_for_backward`

We have mentioned `push_for_backward` will automatically handle multiple versions of a variable. If you are familiar with PyTorch, you may have found the name is similar to PyTorch's `save_for_backward`. Here, versioning is the major difference: `ft.push_for_backward` can be called multiple times on a variable, to save multiple version (or snapshot of it), while the variable can keep changing.

Here is an additional example: a softmax written in a loop form, where we receives a 2-d input, and apply softmax on the second dimension. Again, this is only for demonstration, and there are multiple ways to implement a softmax.

```python
import freetensor as ft
import torch

n = 4

@ft.optimize  # Set verbose=1 to see the code
@ft.grad(requires=['x'], provides=[ft.Return()], attach_backward=True)
def test(x: ft.Var[(n, n), "float32"]):
    y = ft.empty((n, n), "float32")
    for i in range(n):
        # Mark the range that you want to provide graident for, with `StmtRange`
        with ft.StmtRange() as rng:
            # `m`, `e` and `s` are local to `i`
            m = ft.reduce_max(x[i], axes=[-1])
            e = ft.exp(x[i] - m)
            s = ft.reduce_sum(e, axes=[-1])
            y[i] = e / s

            # Call `push_for_backward` so we can use forward values in backward
            e_now = ft.push_for_backward(e)
            s_now = ft.push_for_backward(s)
            y_now = ft.push_for_backward(y)
        # Define gradient in `UserGrad`
        with ft.UserGrad(x, y, stmt_range=rng) as (dzdx, dzdy):
            # Retrieve forward value from `y_now`, NOT `y`
            dzds = -ft.reduce_sum(dzdy[i] * y_now[i], axes=[-1]) / s_now
            dzde = dzdy[i] / s_now + dzds
            dzdx[i] += dzde * e_now  # Use `+=` here
    return y

# Check forward result
x = torch.rand(n, n, dtype=torch.float32)
x.requires_grad = True
y_ft = test(x).torch()
y_torch = torch.softmax(x, axis=-1)
assert torch.all(torch.isclose(y_ft, y_torch))

# Check backward result
y_torch.grad = dzdy = torch.rand(n, n, dtype=torch.float32)
input_grads = test.input_name_to_gradient_name
output_grads = test.output_name_to_gradient_name
dzdx_ft = test.backward(**{output_grads[ft.Return()]: dzdy}).torch()
y_torch.backward(y_torch.grad)
dzdx_torch = x.grad
assert torch.all(torch.isclose(dzdx_ft, dzdx_torch, 1e-4, 1e-7))
```

Here our gradient scope is inside a loop, where `m`, `e` and `s` are local to the loop iteration. When we load the value from their `push_for_backward` handlers, we get the version of value at the exact iteration we need.
