# Optimize a Program with Schedules

[TOC]

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
    #! label: Li  # <-- 1. Label the loop as Li
    for i in range(n):
        y[i] = a[i] + b[i]
    return y

y = test(np.array([1, 2, 3, 4], dtype="int32"),
         np.array([2, 3, 4, 5], dtype="int32")).numpy()
print(y)
```

Here is an example of a parallel vector addition executed with OpenMP multithreading. Each element is computed by one thread. To achieve this, there are two steps:

1. Label the loop to be parallelized with a `#! label:` comment. Here `label` refers to label of an AST node, which is not required to be unique.
2. Apply a `parallelize` schedule to `Li` in the `schedule_callback` argument to `optimize`; since the `Li` label is unambiguous here, the only `Li` loop is selectd and parallelized.

And you are done. You can have a look at the generated OpenMP multithreaded code by setting `verbose=1`.

Parameter `s` in `schedule_callback` is a [`Schedule`](../../api/#freetensor.core.schedule.Schedule) object. Besides `parallelize`, there are more supported scheduling primitives.

If you are using the [`@optimize_to_pytorch` integration](../first-program/#copy-free-interface-fromto-pytorch), you need to set schedules for the forward pass and the backward pass separately.

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

One important thing is to track labels of the loops, because the labels will change after schedules. You get labels (to be precise, IDs, which is can be looked-up by labels) of new loops generated from one schedule from its return values (`outer` and `inner` in this case), and pass them to a next schedule.

## Specify What to Schedule by Selectors

In the example above, we label a loop `Li` and apply schedules on it. It is straight-forward in a tiny example, but as programs grow, it often gets hard to track each statement by a unique label, especially there are inlined function calls. To make things easy, FreeTensor supports specifying a statement by a selector, written in the following rules:

1. A label is a selector. E.g., `Li` matches a statement with a label `Li`.
2. (For debugging only) A numerical ID is also a selector. E.g., `#31`.
3. A node type surrounded in angle brackets (`<>`) is also a selector. E.g., `<For>` matches for-loop statements.
4. A selector can be extended to match a new statement produced by a previous schedule. E.g., `$split.0{Li}` matches the outer loop split from the loop `Li`. This is useful when return values from schedules are hard to track. Please refer the [API document](../../api/#freetensor.core.schedule.Schedule) for detailed grammar.
5. Selectors can be combined to match a statement by nesting order. `A<-B` matches a statement `A` DIRECTLY NESTED IN another statement `B`. `A<<-B` matches a statement DIRECTLY or INDIRECTLY nested in another statement `B`. `A<-(B<-)*C` matches a statement `A` DIRECTLY or INDIRECTLY nested in another statement `C` with intermedaite nesting statements satisfying the condition in `B`. `B->A` matches a statement `B` directly OUT OF another statement `A`. `B->>A` and `C->(B->)*A` are alike. (`A`, `B`, `C` can be nested selectors.) Use `<-|` for the root node, and `->|` for a leaf node.
6. Selectors can be combined to match a statement by DFS order. `A<:B` matches a statement `A` DIRECTLY BEFORE another statement `B`. `A<<:B` matches a statement `A` DIRECTLY or INDIRECTLY before another statement `B`. `B:>A` matches a statment `B` directly AFTER another statement `A`. `B:>>A` matches a statement `B` directly or indirectly after another statement `A`.
7. Selectors can be combined to match a statement in a function call. `A<~B` matches a statement `A` DIRECTLY called by a call site `B`. `A<<~B` matches a statement DIRECTLY or INDIRECTLY called by a call site `B`. `A<~(B<~)*C` matches a statement `A` DIRECTLY or INDIRECTLY called by a call site `C` with intermediate call sites satisfying the condition in `B`. (`A`, `B`, `C` can be nested selectors.) Use `<~|` for the root function.
8. All the arrow-like selectors (`<-`, `<~`, `<:`, etc.) are right-associated. For example, `A<-B<-C` matches `A` nested in `B`, where `B` is nested in `C`.
9. All the arrow-like selectors can be used with the first argument omitted. For example, `<-B` matches ALL statements nested in `B`.
10. Selectors can be combined with logical "and" (`&`), "or" (`|`), "not" (`!`) and parentheses. E.g., `Li|Lj` matches a statement labeled `Li` OR `Lj`. `Li&Lj` matches a statement labeled `Li&Lj`.

All schedules support passing selectors.

## Auto Scheduling (Experimental)

Manually scheduling a program requires a lot of efforts. We provide an experimental automatic scheduling functions in [`Schedule`](../../api/#freetensor.core.schedule.Schedule). You can call `s.auto_schedule` to pick schedules fully automatically. `s.auto_schedule` calls other `s.auto_xxxxxx` functions internally, you can also call one or some of them instead. Please note that these auto-scheduling functions are experimental, and their API is subject to changes.
