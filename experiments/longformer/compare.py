import sys
import torch
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dir1> <dir2>")
        exit(-1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    for name in ['y', 'd_q', 'd_k', 'd_v']:
        print(f"Comparing {name}")
        data1 = np.load(f"{dir1}/y.out.npy")
        data2 = np.load(f"{dir2}/y.out.npy")
        assert np.all(np.isclose(data2, data1, 1e-5, 1e-5)), f"{name} differs"
    print("All output matches")
