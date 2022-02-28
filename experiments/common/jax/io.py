import jax
import jax.numpy as jnp


def load_txt(filename: str, dtype: str):
    with open(filename) as f:
        shape = list(map(int, f.readline().split()))
        if dtype.startswith('int'):
            data = list(map(int, f.readline().split()))
        elif dtype.startswith('float'):
            data = list(map(float, f.readline().split()))
        else:
            assert False
        return jnp.array(data, dtype=dtype).reshape(shape)


def store_txt(filename: str, tensor: jnp.array):
    with open(filename, "w") as f:
        f.write(" ".join(map(str, tensor.shape)) + "\n")
        f.write(" ".join(map(str, tensor.flatten())) + "\n")
