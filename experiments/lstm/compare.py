import sys
import torch
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dir1> <dir2>")
        exit(-1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    for name in ['y', 'd_x', 'd_wi', 'd_wc', 'd_wf', 'd_wo', 'd_ui', 'd_uc', 'd_uf', 'd_uo', 'd_bi', 'd_bc', 'd_bf', 'd_bo']:
        print(f"Comparing {name}")
        data1 = np.loadtxt(f"{dir1}/{name}.out")
        data2 = np.loadtxt(f"{dir2}/{name}.out")
        assert np.all(np.isclose(data2, data1, 1e-4, 1e-4)), f"{name} differs"
    print("All output matches")
