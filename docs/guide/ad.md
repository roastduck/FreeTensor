# Automatic Differentiation

Automatic Differentiation (AD) transforms a program to another program that computes the original one's derivative or gradient. FreeTensor supports Reverse-Mode AD, and there is a plan to support Forward-Mode AD in the future.

## Reverse-Mode AD

Suppose there is a program `x -> y -> z -> w` that computes an output `w` from intermediate variables `z` and `y`, and an input variable `x`. Reverse-Mode AD generates a gradient program `dw/dw=1 -> dw/dz -> dw/dy -> dw/dx` that computes `dw/dx` by Chain Rule. `y`, `z` and `w` may be saved in a "tape" when evaluation the original program, to be reused in the gradient one.

If FreeTensor is built with `WITH_PYTORCH=ON`, you can skip this section and turn to the [`@optimize_to_pytorch` integration](../first-program/#copy-free-interface-fromto-pytorch), which integrates seamlessly with PyTorch's autograd mechanism, but will incur some runtime overhead.

Here is an example of Reverse-Mode AD in FreeTensor:

```python
import freetensor as ft
import numpy as np

n = 4

def test(a: ft.Var[(n,), "float32"], b: ft.Var[(n,), "float32"]):
    y = ft.zeros((), "float32")
    for i in range(n):
        y[()] += a[i] * b[i]
    return y

fwd, bwd, input_grads, output_grads = ft.grad(test, ['a', 'b'],
                                              [ft.Return()])
fwd = ft.optimize(fwd)
bwd = ft.optimize(bwd)

a = np.array([0, 1, 2, 3], dtype="float32")
b = np.array([3, 2, 1, 0], dtype="float32")
y = fwd(a, b)
print(y.numpy())
dzdy = np.array(1, dtype='float32')
dzda, dzdb = bwd(**{output_grads[ft.Return()]: dzdy})[input_grads['a'],
                                                      input_grads['b']]
print(dzda.numpy())
print(dzdb.numpy())
```

You need to call [`ft.grad`](../../api/#freetensor.core.autograd.grad) (or the inplace version [`ft.grad_`](../../api/#freetensor.core.autograd.grad_)) to generate a *forward* function and a *backward* function. Please note that the forward function `fwd` is not the same as the original function `test`, because `fwd` may save some intermediate tensors to a global `tape`, and `fwd` must be executed before the backward one `bwd`.

After that, you call `ft.optimize` to optimize and compile the program just as in previous examples, but for both `fwd` and `bwd` this time.

Finally, you execute `fwd` and `bwd`. The parameters and return values of `bwd` are the gradients of `a`, `b` and `y`, which have their own names. To set and get these parameters and return values, you look up for them in two dictionaries `input_grads` and `output_grads` returned from `ft.grad` (in type [`ft.ArgRetDict`](../../api/#freetensor.core.autograd.ArgRetDict). `input_grads` and `output_grads` accept either a name of a parameter, or a special [`ft.Return`](../../api/#freetensor.core.autograd.Return) to specify a return value. When invoking `bwd`, parameters can be set via keyword arguments, and return values can be collect via a bracket (from a special type [`ft.ReturnValuesPack`](../../api/#freetensor.core.driver.ReturnValuesPack)).

Intermediate variables are not always have to be saved to the "tape" from the forward function. If a variable is need in the backward function but not saved, it will be re-computed, which is sometimes even faster than saving it due to better locality. By default, FreeTensor uses heuristics to determine which variable to save. To get better performance, you may want to control which intermediate variables should be saved by setting an optional `tapes` parameter in `ft.grad`. `tapes` can either be a different mode, or a explicit list of AST node IDs of all `VarDef` nodes of the variables you want to save.
