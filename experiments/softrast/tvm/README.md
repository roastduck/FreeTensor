We implement the task in two parts in `part1.py` and `part2.py`, respectively, because we failed to implement it as a single program.

The first part is not tuned, because we failed to tune it either using AutoTVM or using AutoSchedule. This part is implemented using Relay, because Relay provides a predefined implementation, which runs without scheduling.

The second part is implemented with one (for CPU) or two (for GPU, because we failed to implement it as one) custom operator(s), using `tvm.te.compute`, tuned with AutoSchedule.

As the first part is implemented with Relay, while the second part is implemented with `tvm.te.compute`, it is hard to put it together as one program. We run the first part first, save the temporary data into files, load the data into the second part, and finally run the second part. We add the running time of the two parts together as the total running time, where file I/O time is ignored.
