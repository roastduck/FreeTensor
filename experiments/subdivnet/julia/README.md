While an array($x$ in the case, $a$ in hack) was repeatedly indexed by the other array($adj$ in the case, $b$ in hack), the julia program cannot get correct derivative of the former array just on GPU (nothing wrong on CPU).
The program in `experiments/subdivnet/julia/why_julia_gpu_is_wrong` shows the bug.
