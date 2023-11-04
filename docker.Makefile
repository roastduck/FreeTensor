FT_VERSION := $(shell git rev-parse HEAD)

.PHONY: all
all: gcc-minimal-dev clang-minimal-dev cuda-mkl-dev clang-mkl-dev cuda-mkl-pytorch-dev

.PHONY: gcc-minimal-dev
gcc-minimal-dev:
	docker build -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: clang-minimal-dev
clang-minimal-dev:
	docker build -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: cuda-mkl-dev
cuda-mkl-dev:
	docker build -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: clang-mkl-dev
clang-mkl-dev:
	docker build -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .

.PHONY: cuda-mkl-pytorch-dev
cuda-mkl-pytorch-dev:
	docker build -f $@.Dockerfile -t "freetensor:$@-$(FT_VERSION)" .
