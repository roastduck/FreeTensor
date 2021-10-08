from typing import Sequence, Optional

from .. import core
from .common import StaticType
from .conv_shape_utils import *


def conv_(t_X: StaticType,
          t_W: StaticType,
          t_B: Optional[StaticType],
          t_Y: StaticType,
          io_mem,
          idx_dtype="int32",
          auto_pad: str = 'NOTSET',
          dilations: Optional[Sequence[int]] = None,
          group: int = 1,
          kernel_shape: Optional[Sequence[int]] = None,
          pads: Optional[Sequence[int]] = None,
          strides: Optional[Sequence[int]] = None):

    n_spatial_dim = t_X.ndim - 2

    if dilations is None:
        dilations = [1 for i in range(n_spatial_dim)]
    if strides is None:
        strides = [1 for i in range(n_spatial_dim)]
    if pads is None:
        if auto_pad == 'VALID':
            pads = list(zip(*[[0, 0] for i in range(n_spatial_dim)]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_UPPER':
            assert kernel_shape is not None, "SAME_UPPER pad with dynamic kernel_shape is currently not supported"  # TODO
            pads = list(
                zip(*[
                    calc_same_upper_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_LOWER':
            assert kernel_shape is not None, "SAME_UPPER pad with dynamic kernel_shape is currently not supported"  # TODO
            pads = list(
                zip(*[
                    calc_same_lower_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        else:
            assert False, "auto_pad should be set if pads is not specified"

    assert n_spatial_dim == 2, "Currently only 2-D convolution is supported"  # TODO

    if t_B is None:

        @core.inline
        def f_conv2d(X_shape, W_shape, Y_shape, X, W, Y):
            'nid: V_X_shape'
            core.declare_var(X_shape, (4,), idx_dtype, "input",
                             io_mem)  # N * C * H * W
            'nid: V_W_shape'
            core.declare_var(W_shape, (4,), idx_dtype, "input",
                             io_mem)  # C_out * C_in/group * kH * kW
            'nid: V_Y_shape'
            core.declare_var(Y_shape, (4,), idx_dtype, "input",
                             io_mem)  # N * C * H * W
            'nid: V_X'
            core.declare_var(X, X_shape, t_X.elem_type, "input", io_mem)
            'nid: V_W'
            core.declare_var(W, W_shape, t_W.elem_type, "input", io_mem)
            'nid: V_Y'
            core.declare_var(Y, Y_shape, t_Y.elem_type, "output", io_mem)

            # yapf: disable
            'nid: L_n'
            for n in range(X_shape[0]):
                'nid: L_g'
                for g in range(group):
                    'nid: L_c_out'
                    for c_out in range(W_shape[0] // group):
                        'nid: L_h'
                        for h in range(Y_shape[2]):
                            'nid: L_w'
                            for w in range(Y_shape[3]):
                                'nid: init'
                                Y[n, g * (W_shape[0] // group) + c_out, h, w] = 0
                                'nid: L_c_in'
                                for c_in in range(W_shape[1]):
                                    'nid: L_kh'
                                    for kh in range(W_shape[2]):
                                        'nid: L_kw'
                                        for kw in range(W_shape[3]):
                                            # h_in = h * stride + kh * dilation - pad
                                            # w_in = w * stride + kw * dilation - pad
                                            if (
                                                    h * strides[0] + kh * dilations[0] - pads[0] >= 0 and
                                                    h * strides[0] + kh * dilations[0] - pads[0] < X_shape[2] and
                                                    w * strides[1] + kw * dilations[1] - pads[1] >= 0 and
                                                    w * strides[1] + kw * dilations[1] - pads[1] < X_shape[3]):
                                                'nid: compute'
                                                Y[n, g * (W_shape[0] // group) + c_out, h, w] += X[
                                                        n,
                                                        g * W_shape[1] + c_in,
                                                        h * strides[0] + kh * dilations[0] - pads[0],
                                                        w * strides[1] + kw * dilations[1] - pads[1]
                                                ] * W[g * (W_shape[0] // group) + c_out, c_in, kh, kw]
            # yapf: enable

    else:

        @core.inline
        def f_conv2d(X_shape, W_shape, B_shape, Y_shape, X, W, B, Y):
            'nid: V_X_shape'
            core.declare_var(X_shape, (4,), idx_dtype, "input",
                             io_mem)  # N * C * H * W
            'nid: V_W_shape'
            core.declare_var(W_shape, (4,), idx_dtype, "input",
                             io_mem)  # C_out * C_in/group * kH * kW
            'nid: V_B_shape'
            core.declare_var(B_shape, (1,), idx_dtype, "input", io_mem)  # C_out
            'nid: V_Y_shape'
            core.declare_var(Y_shape, (4,), idx_dtype, "input",
                             io_mem)  # N * C * H * W
            'nid: V_X'
            core.declare_var(X, X_shape, t_X.elem_type, "input", io_mem)
            'nid: V_W'
            core.declare_var(W, W_shape, t_W.elem_type, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, t_B.elem_type, "input", io_mem)
            'nid: V_Y'
            core.declare_var(Y, Y_shape, t_Y.elem_type, "output", io_mem)

            # yapf: disable
            'nid: L_n'
            for n in range(X_shape[0]):
                'nid: L_g'
                for g in range(group):
                    'nid: L_c_out'
                    for c_out in range(W_shape[0] // group):
                        'nid: L_h'
                        for h in range(Y_shape[2]):
                            'nid: L_w'
                            for w in range(Y_shape[3]):
                                'nid: init'
                                Y[n, g * (W_shape[0] // group) + c_out, h, w] = B[g * (W_shape[0] // group) + c_out]
                                'nid: L_c_in'
                                for c_in in range(W_shape[1]):
                                    'nid: L_kh'
                                    for kh in range(W_shape[2]):
                                        'nid: L_kw'
                                        for kw in range(W_shape[3]):
                                            # h_in = h * stride + kh * dilation - pad
                                            # w_in = w * stride + kw * dilation - pad
                                            if (
                                                    h * strides[0] + kh * dilations[0] - pads[0] >= 0 and
                                                    h * strides[0] + kh * dilations[0] - pads[0] < X_shape[2] and
                                                    w * strides[1] + kw * dilations[1] - pads[1] >= 0 and
                                                    w * strides[1] + kw * dilations[1] - pads[1] < X_shape[3]):
                                                'nid: compute'
                                                Y[n, g * (W_shape[0] // group) + c_out, h, w] += X[
                                                        n,
                                                        g * W_shape[1] + c_in,
                                                        h * strides[0] + kh * dilations[0] - pads[0],
                                                        w * strides[1] + kw * dilations[1] - pads[1]
                                                ] * W[g * (W_shape[0] // group) + c_out, c_in, kh, kw]
            # yapf: enable

    return f_conv2d


def conv(t_X: StaticType,
         t_W: StaticType,
         t_B: Optional[StaticType],
         t_Y: StaticType,
         io_mem,
         idx_dtype="int32",
         auto_pad: str = 'NOTSET',
         dilations: Optional[Sequence[int]] = None,
         group: int = 1,
         kernel_shape: Optional[Sequence[int]] = None,
         pads: Optional[Sequence[int]] = None,
         strides: Optional[Sequence[int]] = None):

    n_spatial_dim = t_X.ndim - 2

    if dilations is None:
        dilations = [1 for i in range(n_spatial_dim)]
    if strides is None:
        strides = [1 for i in range(n_spatial_dim)]
    if pads is None:
        if auto_pad == 'VALID':
            pads = list(zip(*[[0, 0] for i in range(n_spatial_dim)]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_UPPER':
            assert kernel_shape is not None, "SAME_UPPER pad with dynamic kernel_shape is currently not supported"  # TODO
            pads = list(
                zip(*[
                    calc_same_upper_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_LOWER':
            assert kernel_shape is not None, "SAME_UPPER pad with dynamic kernel_shape is currently not supported"  # TODO
            pads = list(
                zip(*[
                    calc_same_lower_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        else:
            assert False, "auto_pad should be set if pads is not specified"

    assert n_spatial_dim == 2, "Currently only 2-D convolution is supported"  # TODO

    if t_B is None:

        @core.inline
        def f_conv2d(X_shape, W_shape, X, W):
            'nid: V_X_shape'
            core.declare_var(X_shape, (4,), idx_dtype, "input",
                             io_mem)  # N * C * H * W
            'nid: V_W_shape'
            core.declare_var(W_shape, (4,), idx_dtype, "input",
                             io_mem)  # C_out * C_in/group * kH * kW
            'nid: V_X'
            core.declare_var(X, X_shape, t_X.elem_type, "input", io_mem)
            'nid: V_W'
            core.declare_var(W, W_shape, t_W.elem_type, "input", io_mem)

            'nid: V_Y_shape'
            Y_shape = core.create_var((4,), idx_dtype, "output",
                                      io_mem)  # N * C * H * W
            Y_shape[0] = X_shape[0]
            Y_shape[1] = W_shape[0]
            Y_shape[2] = calc_out_size(X_shape[2], dilations[0], W_shape[2],
                                       pads[0], pads[2], strides[0])
            Y_shape[3] = calc_out_size(X_shape[3], dilations[1], W_shape[3],
                                       pads[1], pads[3], strides[1])

            'nid: V_Y'
            Y = core.create_var(Y_shape, t_Y.elem_type, "output", io_mem)

            'nid: recur'
            conv_(t_X, t_W, t_B, t_Y, io_mem, idx_dtype, auto_pad, dilations,
                  group, kernel_shape, pads, strides)(X_shape, W_shape, Y_shape,
                                                      X, W, Y)

            return Y

    else:

        @core.inline
        def f_conv2d(X_shape, W_shape, B_shape, X, W, B):
            'nid: V_X_shape'
            core.declare_var(X_shape, (4,), idx_dtype, "input",
                             io_mem)  # N * C * H * W
            'nid: V_W_shape'
            core.declare_var(W_shape, (4,), idx_dtype, "input",
                             io_mem)  # C_out * C_in/group * kH * kW
            'nid: V_B_shape'
            core.declare_var(B_shape, (1,), idx_dtype, "input", io_mem)  # C_out
            'nid: V_X'
            core.declare_var(X, X_shape, t_X.elem_type, "input", io_mem)
            'nid: V_W'
            core.declare_var(W, W_shape, t_W.elem_type, "input", io_mem)
            'nid: V_B'
            core.declare_var(B, B_shape, t_B.elem_type, "input", io_mem)

            'nid: V_Y_shape'
            Y_shape = core.create_var((4,), idx_dtype, "output",
                                      io_mem)  # N * C * H * W
            Y_shape[0] = X_shape[0]
            Y_shape[1] = W_shape[0]
            Y_shape[2] = calc_out_size(X_shape[2], dilations[0], W_shape[2],
                                       pads[0], pads[2], strides[0])
            Y_shape[3] = calc_out_size(X_shape[3], dilations[1], W_shape[3],
                                       pads[1], pads[3], strides[1])

            'nid: V_Y'
            Y = core.create_var(Y_shape, t_Y.elem_type, "output", io_mem)

            'nid: recur'
            conv_(t_X, t_W, t_B, t_Y, io_mem, idx_dtype, auto_pad, dilations,
                  group, kernel_shape, pads, strides)(X_shape, W_shape, B_shape,
                                                      Y_shape, X, W, B, Y)

            return Y

    return f_conv2d
