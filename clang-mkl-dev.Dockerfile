FROM ubuntu:22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ python3 python3-dev python3-pip python3-venv cmake make ninja-build \
    autoconf automake libtool openjdk-11-jdk libgmp-dev wget xz-utils libtinfo5 \
    libmkl-dev

# Install Clang 16
WORKDIR /utils/clang
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz \
    && tar -xf clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
ENV PATH=/utils/clang/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/:$PATH
ENV LD_LIBRARY_PATH=/utils/clang/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04/lib:$LD_LIBRARY_PATH
ENV CC=clang
ENV CXX=clang++

RUN pip3 install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple pip # We need `-C` from pip

WORKDIR /opt/freetensor
COPY . .
RUN PY_BUILD_CMAKE_VERBOSE=1 pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e . \
    -C--local=with-mkl.toml

# `pip3 install` only installs `freetensor_ffi`. We also need other stuffs installed.
# (FIXME: install everything using `pip3 install` and properly set the paths)
RUN cmake --install build/cp310-cp310-linux_x86_64

WORKDIR /workspace
