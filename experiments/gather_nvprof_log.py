import sys


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


filename = sys.argv[1]

kernels = 0
dram_bytes = 0
l2_bytes = 0
flops = 0
flops_mul = 0
flops_fma = 0

for line in open(filename):
    cols = line.split()
    if len(cols) >= 6:
        if cols[1] == 'dram_read_bytes':
            dram_bytes += num(cols[0]) * num(cols[-1])
            kernels += num(cols[0])
        if cols[1] == 'dram_write_bytes':
            dram_bytes += num(cols[0]) * num(cols[-1])
        if cols[1] == 'l2_global_load_bytes':
            l2_bytes += num(cols[0]) * num(cols[-1])
        if cols[1] == 'l2_local_load_bytes':
            l2_bytes += num(cols[0]) * num(cols[-1])
        if cols[1] == 'l2_global_atomic_store_bytes':
            l2_bytes += num(cols[0]) * num(cols[-1])
        if cols[1] == 'l2_local_global_store_bytes':
            l2_bytes += num(cols[0]) * num(cols[-1])
        if cols[1] == 'flop_count_sp':
            flops += num(cols[0]) * num(cols[-1])
        if cols[1] == 'flop_count_sp_mul':
            flops_mul += num(cols[0]) * num(cols[-1])
        if cols[1] == 'flop_count_sp_fma':
            flops_fma += num(cols[0]) * num(cols[-1])

print("Kernels: ", kernels)
print("DRAM bytes: ", dram_bytes)
print("L2 bytes: ", l2_bytes)
print("FLOP: ", flops)
print("FLOP (MUL): ", flops_mul)
print("FLOP (FMA): ", flops_fma)
