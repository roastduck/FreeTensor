FROM nvcr.io/nvidia/pytorch:23.05-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-05.html#rel-23-05
# CUDA 12.1.1
# PyTorch 2.0.0

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ python3 python3-dev python3-pip python3-venv cmake make ninja-build \
    autoconf automake libtool openjdk-11-jdk libgmp-dev \
    libmkl-dev

RUN pip3 install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple pip # We need `-C` from pip

WORKDIR /opt/freetensor
COPY . .

# We use --no-build-isolation to disable building in a virtual environment because we need
# PyTorch from the system. But this requires us to install build dependencies on our own
RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "py-build-cmake~=0.1.8"

RUN PY_BUILD_CMAKE_VERBOSE=1 python3 -m pip install --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e . \
    -C--local=with-cuda.toml \
    -C--local=with-mkl.toml \
    -C--local=with-pytorch.toml

# `pip3 install` only installs `freetensor_ffi`. We also need other stuffs installed.
# (FIXME: install everything using `pip3 install` and properly set the paths)
RUN cmake --install build/cp310-cp310-linux_x86_64

WORKDIR /workspace
