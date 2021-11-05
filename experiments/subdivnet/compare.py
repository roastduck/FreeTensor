import sys
import torch
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dir1> <dir2>")
        exit(-1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    for name in ['y', 'd_x', 'd_w0', 'd_w1', 'd_w2', 'd_w3']:
        print(f"Comparing {name}")
        data1 = np.loadtxt(f"{dir1}/y.out")
        data2 = np.loadtxt(f"{dir2}/y.out")
        assert np.all(np.isclose(data2, data1)), f"{name} differs"
    print("All output matches")
