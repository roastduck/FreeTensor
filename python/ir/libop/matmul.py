from typing import Optional

from .. import core


def gemm_(io_mem,
          has_bias: bool = False,
          trans_A: bool = False,
          trans_B: bool = False,
          alpha: float = 1.0,
          beta: float = 1.0):

    if not has_bias:

        @core.inline
        def f_gemm(A, B, Y):
            if not trans_A:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A.shape(0)):
                        'nid: L_j'
                        for j in range(B.shape(1)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(1)):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[k, j]
                            'nid: bias'
                            Y[i, j] *= alpha
                else:
                    'nid: L_i'
                    for i in range(A.shape(0)):
                        'nid: L_j'
                        for j in range(B.shape(0)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(1)):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[j, k]
                            'nid: bias'
                            Y[i, j] *= alpha
            else:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A.shape(1)):
                        'nid: L_j'
                        for j in range(B.shape(1)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(0)):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[k, j]
                            'nid: bias'
                            Y[i, j] *= alpha
                else:
                    'nid: L_i'
                    for i in range(A.shape(1)):
                        'nid: L_j'
                        for j in range(B.shape(0)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(0)):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[j, k]
                            'nid: bias'
                            Y[i, j] *= alpha

    else:

        @core.inline
        def f_gemm(A, B, C, Y):
            if not trans_A:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A.shape(0)):
                        'nid: L_j'
                        for j in range(B.shape(1)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(1)):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[k, j]
                            if C.ndim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif C.ndim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C.shape(0)]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C.shape(0), j % C.shape(1)]
                else:
                    'nid: L_i'
                    for i in range(A.shape(0)):
                        'nid: L_j'
                        for j in range(B.shape(0)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(1)):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[j, k]
                            if C.ndim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif C.ndim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C.shape(0)]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C.shape(0), j % C.shape(1)]
            else:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A.shape(1)):
                        'nid: L_j'
                        for j in range(B.shape(1)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(0)):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[k, j]
                            if C.ndim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif C.ndim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C.shape(0)]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C.shape(0), j % C.shape(1)]
                else:
                    'nid: L_i'
                    for i in range(A.shape(1)):
                        'nid: L_j'
                        for j in range(B.shape(0)):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A.shape(0)):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[j, k]
                            if C.ndim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif C.ndim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C.shape(0)]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C.shape(0), j % C.shape(1)]

    return f_gemm


def gemm(io_mem,
         has_bias: bool = False,
         trans_A: bool = False,
         trans_B: bool = False,
         alpha: float = 1.0,
         beta: float = 1.0):

    def comp_shape(A, B):
        if not trans_A:
            if not trans_B:
                return [A.shape(0), B.shape(1)]
            else:
                return [A.shape(0), B.shape(0)]
        else:
            if not trans_B:
                return [A.shape(1), B.shape(1)]
            else:
                return [A.shape(1), B.shape(0)]

    if not has_bias:

        @core.inline
        def f_gemm(A, B):
            Y = core.create_var(comp_shape(A,
                                           B), core.up_cast(A.dtype, B.dtype),
                                "output", io_mem)
            'nid: recur'
            gemm_(io_mem, has_bias, trans_A, trans_B, alpha, beta)(A, B, Y)
            return Y

    else:

        @core.inline
        def f_gemm(A, B, C):
            Y = core.create_var(comp_shape(A,
                                           B), core.up_cast(A.dtype, B.dtype),
                                "output", io_mem)
            'nid: recur'
            gemm_(io_mem, has_bias, trans_A, trans_B, alpha, beta)(A, B, Y)
            return Y

    return f_gemm
