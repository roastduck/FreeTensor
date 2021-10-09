from typing import Sequence, Optional

from .. import core
from .conv_shape_utils import *


def conv_(io_mem,
          has_bias: bool = False,
          auto_pad: str = 'NOTSET',
          dilations: Optional[Sequence[int]] = None,
          group: int = 1,
          kernel_shape: Optional[Sequence[int]] = None,
          pads: Optional[Sequence[int]] = None,
          strides: Optional[Sequence[int]] = None):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

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

    if not has_bias:

        @core.inline
        def f_conv2d(X, W, Y):
            # yapf: disable
            'nid: L_n'
            for n in range(X.shape(0)):
                'nid: L_g'
                for g in range(group):
                    'nid: L_c_out'
                    for c_out in range(W.shape(0) // group):
                        'nid: L_h'
                        for h in range(Y.shape(2)):
                            'nid: L_w'
                            for w in range(Y.shape(3)):
                                'nid: init'
                                Y[n, g * (W.shape(0) // group) + c_out, h, w] = 0
                                'nid: L_c_in'
                                for c_in in range(W.shape(1)):
                                    'nid: L_kh'
                                    for kh in range(W.shape(2)):
                                        'nid: L_kw'
                                        for kw in range(W.shape(3)):
                                            # h_in = h * stride + kh * dilation - pad
                                            # w_in = w * stride + kw * dilation - pad
                                            if (
                                                    h * strides[0] + kh * dilations[0] - pads[0] >= 0 and
                                                    h * strides[0] + kh * dilations[0] - pads[0] < X.shape(2) and
                                                    w * strides[1] + kw * dilations[1] - pads[1] >= 0 and
                                                    w * strides[1] + kw * dilations[1] - pads[1] < X.shape(3)):
                                                'nid: compute'
                                                Y[n, g * (W.shape(0) // group) + c_out, h, w] += X[
                                                        n,
                                                        g * W.shape(1) + c_in,
                                                        h * strides[0] + kh * dilations[0] - pads[0],
                                                        w * strides[1] + kw * dilations[1] - pads[1]
                                                ] * W[g * (W.shape(0) // group) + c_out, c_in, kh, kw]
            # yapf: enable

    else:

        @core.inline
        def f_conv2d(X, W, B, Y):
            # yapf: disable
            'nid: L_n'
            for n in range(X.shape(0)):
                'nid: L_g'
                for g in range(group):
                    'nid: L_c_out'
                    for c_out in range(W.shape(0) // group):
                        'nid: L_h'
                        for h in range(Y.shape(2)):
                            'nid: L_w'
                            for w in range(Y.shape(3)):
                                'nid: init'
                                Y[n, g * (W.shape(0) // group) + c_out, h, w] = B[g * (W.shape(0) // group) + c_out]
                                'nid: L_c_in'
                                for c_in in range(W.shape(1)):
                                    'nid: L_kh'
                                    for kh in range(W.shape(2)):
                                        'nid: L_kw'
                                        for kw in range(W.shape(3)):
                                            # h_in = h * stride + kh * dilation - pad
                                            # w_in = w * stride + kw * dilation - pad
                                            if (
                                                    h * strides[0] + kh * dilations[0] - pads[0] >= 0 and
                                                    h * strides[0] + kh * dilations[0] - pads[0] < X.shape(2) and
                                                    w * strides[1] + kw * dilations[1] - pads[1] >= 0 and
                                                    w * strides[1] + kw * dilations[1] - pads[1] < X.shape(3)):
                                                'nid: compute'
                                                Y[n, g * (W.shape(0) // group) + c_out, h, w] += X[
                                                        n,
                                                        g * W.shape(1) + c_in,
                                                        h * strides[0] + kh * dilations[0] - pads[0],
                                                        w * strides[1] + kw * dilations[1] - pads[1]
                                                ] * W[g * (W.shape(0) // group) + c_out, c_in, kh, kw]
            # yapf: enable

    return f_conv2d


def conv(io_mem,
         has_bias: bool = False,
         auto_pad: str = 'NOTSET',
         dilations: Optional[Sequence[int]] = None,
         group: int = 1,
         kernel_shape: Optional[Sequence[int]] = None,
         pads: Optional[Sequence[int]] = None,
         strides: Optional[Sequence[int]] = None):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

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

    if not has_bias:

        @core.inline
        def f_conv2d(X, W):
            'nid: V_Y'
            Y = core.create_var([
                X.shape(0),
                W.shape(0),
                calc_out_size(X.shape(2), dilations[0], W.shape(2), pads[0],
                              pads[2], strides[0]),
                calc_out_size(X.shape(3), dilations[1], W.shape(3), pads[1],
                              pads[3], strides[1])
            ], core.up_cast(X.dtype, W.dtype), "output", io_mem)

            'nid: recur'
            conv_(io_mem, has_bias, auto_pad, dilations, group, kernel_shape,
                  pads, strides)(X, W, Y)

            return Y

    else:

        @core.inline
        def f_conv2d(X, W, B):
            'nid: V_Y'
            Y = core.create_var([
                X.shape(0),
                W.shape(0),
                calc_out_size(X.shape(2), dilations[0], W.shape(2), pads[0],
                              pads[2], strides[0]),
                calc_out_size(X.shape(3), dilations[1], W.shape(3), pads[1],
                              pads[3], strides[1])
            ], core.up_cast(core.up_cast(X.dtype, W.dtype), B.dtype), "output",
                                io_mem)

            'nid: recur'
            conv_(io_mem, has_bias, auto_pad, dilations, group, kernel_shape,
                  pads, strides)(X, W, B, Y)

            return Y

    return f_conv2d
