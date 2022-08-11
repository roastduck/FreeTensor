# Optimize a Program with Schedules

Oftentimes, only compiling your programs to native code is not enough, and you need further optimizations. This can be done by applying "schedules" (explicit program transformations) to you program.

## Example: Parallel Vector addition

```python
import freetensor as ft
import numpy as np

n = 4

# Add verbose=1 to see the resulting native code
@ft.optimize(schedule_callback=lambda s: s.parallelize('Li', 'openmp')
            )  # <-- 2. Apply the schedule
def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
    y = ft.empty((n,), "int32")
    #! label: Li  # <-- 1. Name the loop as Li
    for i in range(n):
        y[i] = a[i] + b[i]
    return y

y = test(np.array([1, 2, 3, 4], dtype="int32"),
         np.array([2, 3, 4, 5], dtype="int32")).numpy()
print(y)
```

Here is an example of a parallel vector addition executed with OpenMP multithreading. Each element is computed by one thread. To achieve this, there are two steps:

1. Name the loop to be parallelized with a `#! label:` comment. Here `label` refers to label of an AST node, which is not required to be unique.
2. Apply a `parallelize` schedule to `Li` in the `schedule_callback` argument to `optimize`; since the `Li` label is unambiguous here, the only `Li` loop is selectd and parallelized.

And you are done. You can have a look at the generated OpenMP multithreaded code by setting `verbose=1`.

Parameter `s` in `schedule_callback` is a [`Schedule`](../../api/#freetensor.core.schedule.Schedule) object. Besides `parallelize`, there are more supported scheduling primitives.

## Combining Multiple Schdules

Some optimizations can be done by applying multiple schedules. For example, a tiled matrix-multiplication can be done by first `split` the loops, then `reorder` them, and finally apply `cache`s to create tile tensors.

In order to demonstrate the idea, we show a simplier example here: still a vector addtion, but with the loop `split` and only the outer one `parallelize`d. Please note that this is an example only for demonstration. Usually you do not need it because OpenMP has its own "schedule(static)" for parallelized loops.

```python
import freetensor as ft
import numpy as np

n = 1024

def sch(s):
    outer, inner = s.split('Li', 32)
    s.parallelize(outer, 'openmp')

# Set verbose=1 to see the resulting native code
# Set verbose=2 to see the code after EVERY schedule
@ft.optimize(schedule_callback=sch)
def test(a: ft.Var[(n,), "int32"], b: ft.Var[(n,), "int32"]):
    y = ft.empty((n,), "int32")
    #! label: Li
    for i in range(n):
        y[i] = a[i] + b[i]
    return y

y = test(np.array(np.arange(1024), dtype="int32"),
         np.array(np.arange(1024), dtype="int32")).numpy()
print(y)
```

One important thing is to track names of the loops, because the names will change after schedules. You get names of new loops generated from one schedule from its return values (`outer` and `inner` in this case), and pass them to a next schedule.

## Auto Scheduling (Experimental)

Manually scheduling a program requires a lot of efforts. We provide an experimental automatic scheduling functions in [`Schedule`](../../api/#freetensor.core.schedule.Schedule). You can call `s.auto_schedule` to pick schedules fully automatically. `s.auto_schedule` calls other `s.auto_xxxxxx` functions internally, you can also call one or some of them instead. Please note that these auto-scheduling functions are experimental, and their API is subject to changes.
