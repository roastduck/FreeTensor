from typing import Optional

from .. import core
from .common import StaticType


def gemm_(t_A: StaticType,
          t_B: StaticType,
          t_C: Optional[StaticType],
          t_Y: StaticType,
          io_mem,
          idx_dtype="int32",
          trans_A: bool = False,
          trans_B: bool = False,
          alpha: float = 1.0,
          beta: float = 1.0):

    if t_C is None:

        @core.transform
        def f_gemm(A_shape, B_shape, Y_shape, A, B, Y):
            'nid: V_A_shape'
            core.declare_var(A_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_B_shape'
            core.declare_var(B_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_Y_shape'
            core.declare_var(Y_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_A'
            core.declare_var(A, A_shape, t_A.elem_type, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, t_B.elem_type, "input", io_mem)
            'nid: V_Y'
            core.declare_var(Y, Y_shape, t_Y.elem_type, "output", io_mem)

            if not trans_A:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[k, j]
                            'nid: bias'
                            Y[i, j] *= alpha
                else:
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[j, k]
                            'nid: bias'
                            Y[i, j] *= alpha
            else:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[k, j]
                            'nid: bias'
                            Y[i, j] *= alpha
                else:
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[j, k]
                            'nid: bias'
                            Y[i, j] *= alpha

    else:

        n_bias_dim = t_C.ndim
        assert n_bias_dim <= 2

        @core.transform
        def f_gemm(A_shape, B_shape, C_shape, Y_shape, A, B, C, Y):
            'nid: V_A_shape'
            core.declare_var(A_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_B_shape'
            core.declare_var(B_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_C_shape'
            core.declare_var(C_shape, (n_bias_dim,), idx_dtype, "input", io_mem)
            'nid: V_Y_shape'
            core.declare_var(Y_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_A'
            core.declare_var(A, A_shape, t_A.elem_type, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, t_B.elem_type, "input", io_mem)
            'nid: V_C'
            core.declare_var(C, C_shape, t_C.elem_type, "input", io_mem)
            'nid: V_Y'
            core.declare_var(Y, Y_shape, t_Y.elem_type, "output", io_mem)

            if not trans_A:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[k, j]
                            if n_bias_dim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C_shape[0]]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C_shape[0], j % C_shape[1]]
                else:
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += A[i, k] * B[j, k]
                            if n_bias_dim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C_shape[0]]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C_shape[0], j % C_shape[1]]
            else:
                if not trans_B:
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[k, j]
                            if n_bias_dim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C_shape[0]]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C_shape[0], j % C_shape[1]]
                else:
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += A[k, i] * B[j, k]
                            if n_bias_dim == 0:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: bias'
                                Y[i,
                                  j] = alpha * Y[i,
                                                 j] + beta * C[j % C_shape[0]]
                            else:
                                'nid: bias'
                                Y[i, j] = alpha * Y[i, j] + beta * C[
                                    i % C_shape[0], j % C_shape[1]]

    return f_gemm


def gemm(t_A: StaticType,
         t_B: StaticType,
         t_C: Optional[StaticType],
         t_Y: StaticType,
         io_mem,
         idx_dtype="int32",
         trans_A: bool = False,
         trans_B: bool = False,
         alpha: float = 1.0,
         beta: float = 1.0):

    if t_C is None:

        @core.transform
        def f_gemm(A_shape, B_shape, A, B):
            'nid: V_A_shape'
            core.declare_var(A_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_B_shape'
            core.declare_var(B_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_A'
            core.declare_var(A, A_shape, t_A.elem_type, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, t_B.elem_type, "input", io_mem)
            'nid: V_Y_shape'
            Y_shape = core.create_var((2,), idx_dtype, "output", io_mem)
            if not trans_A:
                if not trans_B:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[1]
                else:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[0]
            else:
                if not trans_B:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[1]
                else:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[0]
            'nid: V_Y'
            Y = core.create_var(Y_shape, t_Y.elem_type, "output", io_mem)
            'nid: recur'
            gemm_(t_A, t_B, t_C, t_Y, io_mem, idx_dtype, trans_A, trans_B,
                  alpha, beta)(A_shape, B_shape, Y_shape, A, B, Y)
            return Y

    else:

        n_bias_dim = t_C.ndim
        assert n_bias_dim <= 2

        @core.transform
        def f_gemm(A_shape, B_shape, C_shape, A, B, C):
            'nid: V_A_shape'
            core.declare_var(A_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_B_shape'
            core.declare_var(B_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_C_shape'
            core.declare_var(C_shape, (n_bias_dim,), idx_dtype, "input", io_mem)
            'nid: V_A'
            core.declare_var(A, A_shape, t_A.elem_type, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, t_B.elem_type, "input", io_mem)
            'nid: V_C'
            core.declare_var(C, C_shape, t_C.elem_type, "input", io_mem)
            'nid: V_Y_shape'
            Y_shape = core.create_var((2,), idx_dtype, "output", io_mem)
            if not trans_A:
                if not trans_B:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[1]
                else:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[0]
            else:
                if not trans_B:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[1]
                else:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[0]
            'nid: V_Y'
            Y = core.create_var(Y_shape, t_Y.elem_type, "output", io_mem)
            'nid: recur'
            gemm_(t_A, t_B, t_C, t_Y, io_mem, idx_dtype, trans_A, trans_B,
                  alpha, beta)(A_shape, B_shape, Y_shape, A, B, Y)
            return Y

    return f_gemm
