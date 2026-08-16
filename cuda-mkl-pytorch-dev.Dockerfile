# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ARG PYTHON_EXTRAS=""

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_CACHE_DIR=/root/.cache/pip \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=10G

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ cmake make ninja-build ccache \
        autoconf automake libtool openjdk-11-jdk libgmp-dev \
        libmkl-dev

# The image provides PyTorch in its Python environment. FreeTensor's PyTorch build must
# use that same environment, so build isolation is disabled below and build-time Python
# requirements are installed explicitly.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        "py-build-cmake~=0.5.0" \
        importlib_metadata \
        z3-solver \
        setuptools

WORKDIR /opt/freetensor
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    install_target="."; \
    if [ -n "${PYTHON_EXTRAS}" ]; then install_target=".[${PYTHON_EXTRAS}]"; fi; \
    PY_BUILD_CMAKE_VERBOSE=1 python3 -m pip install --no-build-isolation \
        -v -e "${install_target}" \
        -C--local=with-cuda.toml \
        -C--local=with-mkl.toml \
        -C--local=with-pytorch.toml

WORKDIR /workspace
