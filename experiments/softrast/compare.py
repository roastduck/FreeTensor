import sys
import numpy as np

sys.path.append('..')
from common.numpy.io import load_txt

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dir1> <dir2>")
        exit(-1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    for name in ['y', 'd_vertices']:
        print(f"Comparing {name}")
        data1 = load_txt(f"{dir1}/{name}.out", "float32")
        data2 = load_txt(f"{dir2}/{name}.out", "float32")
        assert np.all(np.isclose(data2, data1, 5e-2, 5e-3)), f"{name} differs"
    print("All output matches")
