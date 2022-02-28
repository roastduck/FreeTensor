import sys
import numpy as np

sys.path.append('..')
from common.numpy.io import load_txt

if __name__ == '__main__':
    if len(sys.argv) not in range(3, 5):
        print(f"Usage: {sys.argv[0]} <dir1> <dir2> [--infer-only]")
        print(
            "--infer-only: Some baselines does not support computing gradients w.r.t. a tensor, "
            "so we sum the output as a scalar. Pass --infer-only for these baselines so we do not "
            "check to gradient results")
        exit(-1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    to_check = ['y']
    if '--infer-only' not in sys.argv:
        to_check += ['d_vertices']

    for name in to_check:
        print(f"Comparing {name}")
        data1 = load_txt(f"{dir1}/{name}.out", "float32")
        data2 = load_txt(f"{dir2}/{name}.out", "float32")
        assert np.all(np.isclose(data2, data1, 5e-2, 5e-3)), f"{name} differs"
    print("All output matches")
