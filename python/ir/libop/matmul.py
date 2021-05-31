from typing import Optional

from .. import core


def gemm(io_mem,
         data_dtype="float32",
         idx_dtype="int32",
         trans_A: bool = False,
         trans_B: bool = False,
         alpha: float = 1.0,
         beta: float = 1.0,
         with_bias: bool = False,
         n_bias_dim: Optional[int] = None):

    if not with_bias:

        @core.transform
        def f_gemm(A_shape, B_shape, Y_shape, A, B, Y):
            'nid: V_A_shape'
            core.declare_var(A_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_B_shape'
            core.declare_var(B_shape, (2,), idx_dtype, "input", io_mem)
            'nid: V_Y_shape'
            core.declare_var(Y_shape, (2,), idx_dtype, "output", io_mem)
            'nid: V_A'
            core.declare_var(A, A_shape, data_dtype, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, data_dtype, "input", io_mem)
            'nid: V_Y'
            core.declare_var(Y, Y_shape, data_dtype, "output", io_mem)

            if not trans_A:
                if not trans_B:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[1]
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += alpha * A[i, k] * B[k, j]
                else:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[0]
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += alpha * A[i, k] * B[j, k]
            else:
                if not trans_B:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[1]
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += alpha * A[k, i] * B[k, j]
                else:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[0]
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            'nid: init'
                            Y[i, j] = 0
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += alpha * A[k, i] * B[j, k]

    else:

        assert n_bias_dim is not None
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
            core.declare_var(Y_shape, (2,), idx_dtype, "output", io_mem)
            'nid: V_A'
            core.declare_var(A, A_shape, data_dtype, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, data_dtype, "input", io_mem)
            'nid: V_C'
            core.declare_var(C, C_shape, data_dtype, "input", io_mem)
            'nid: V_Y'
            core.declare_var(Y, Y_shape, data_dtype, "output", io_mem)

            if not trans_A:
                if not trans_B:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[1]
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            if n_bias_dim == 0:
                                'nid: init'
                                Y[i, j] = beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: init'
                                Y[i, j] = beta * C[j % C_shape[0]]
                            else:
                                'nid: init'
                                Y[i,
                                  j] = beta * C[i % C_shape[0], j % C_shape[1]]
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += alpha * A[i, k] * B[k, j]
                else:
                    Y_shape[0] = A_shape[0]
                    Y_shape[1] = B_shape[0]
                    'nid: L_i'
                    for i in range(A_shape[0]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            if n_bias_dim == 0:
                                'nid: init'
                                Y[i, j] = beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: init'
                                Y[i, j] = beta * C[j % C_shape[0]]
                            else:
                                'nid: init'
                                Y[i,
                                  j] = beta * C[i % C_shape[0], j % C_shape[1]]
                            'nid: L_k'
                            for k in range(A_shape[1]):
                                'nid: compute'
                                Y[i, j] += alpha * A[i, k] * B[j, k]
            else:
                if not trans_B:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[1]
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[1]):
                            if n_bias_dim == 0:
                                'nid: init'
                                Y[i, j] = beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: init'
                                Y[i, j] = beta * C[j % C_shape[0]]
                            else:
                                'nid: init'
                                Y[i,
                                  j] = beta * C[i % C_shape[0], j % C_shape[1]]
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += alpha * A[k, i] * B[k, j]
                else:
                    Y_shape[0] = A_shape[1]
                    Y_shape[1] = B_shape[0]
                    'nid: L_i'
                    for i in range(A_shape[1]):
                        'nid: L_j'
                        for j in range(B_shape[0]):
                            if n_bias_dim == 0:
                                'nid: init'
                                Y[i, j] = beta * C[()]
                            elif n_bias_dim == 1:
                                'nid: init'
                                Y[i, j] = beta * C[j % C_shape[0]]
                            else:
                                'nid: init'
                                Y[i,
                                  j] = beta * C[i % C_shape[0], j % C_shape[1]]
                            'nid: L_k'
                            for k in range(A_shape[0]):
                                'nid: compute'
                                Y[i, j] += alpha * A[k, i] * B[j, k]

    return f_gemm
