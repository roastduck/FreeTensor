DOCKER_BUILDKIT ?= 1
PYTHON_EXTRAS ?=

FT_VERSION := $(shell git rev-parse HEAD)

.PHONY: all
all: gcc-minimal-dev clang-minimal-dev cuda-mkl-dev clang-mkl-dev cuda-mkl-pytorch-dev

.PHONY: gcc-minimal-dev
gcc-minimal-dev:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg PYTHON_EXTRAS="$(PYTHON_EXTRAS)" -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: clang-minimal-dev
clang-minimal-dev:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg PYTHON_EXTRAS="$(PYTHON_EXTRAS)" -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: cuda-mkl-dev
cuda-mkl-dev:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg PYTHON_EXTRAS="$(PYTHON_EXTRAS)" -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: clang-mkl-dev
clang-mkl-dev:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg PYTHON_EXTRAS="$(PYTHON_EXTRAS)" -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: cuda-mkl-pytorch-dev
cuda-mkl-pytorch-dev:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg PYTHON_EXTRAS="$(PYTHON_EXTRAS)" -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .
