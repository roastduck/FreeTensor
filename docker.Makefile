FT_VERSION := $(shell git rev-parse HEAD)

.PHONY: all
all: minimal-dev cuda-mkl-dev cuda-mkl-pytorch-dev

.PHONY: minimal-dev
minimal-dev:
	docker build -f minimal-dev.Dockerfile -t "freetensor:minimal-dev-$(FT_VERSION)" .

.PHONY: cuda-mkl-dev
cuda-mkl-dev:
	docker build -f cuda-mkl-dev.Dockerfile -t "freetensor:cuda-mkl-dev-$(FT_VERSION)" .

.PHONY: cuda-mkl-pytorch-dev
cuda-mkl-pytorch-dev:
	docker build -f cuda-mkl-pytorch-dev.Dockerfile -t "freetensor:cuda-mkl-pytorch-dev-$(FT_VERSION)" .
