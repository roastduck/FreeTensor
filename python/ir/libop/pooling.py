from typing import Sequence, Optional

from .. import core
from .conv_shape_utils import *


def max_pool_(io_mem,
              auto_pad: str = 'NOTSET',
              dilations: Optional[Sequence[int]] = None,
              kernel_shape: Sequence[int] = None,
              pads: Optional[Sequence[int]] = None,
              strides: Optional[Sequence[int]] = None):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    # TODO: ceil_mode
    # TODO: return_indices

    if dilations is None:
        dilations = [1 for i in range(n_spatial_dim)]
    if strides is None:
        # NOTE: strides default to 1 in ONNX, while default to kernel_shape in PyTorch
        strides = [1 for i in range(n_spatial_dim)]
    if pads is None:
        if auto_pad == 'VALID':
            pads = list(zip(*[[0, 0] for i in range(n_spatial_dim)]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_UPPER':
            pads = list(
                zip(*[
                    calc_same_upper_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_LOWER':
            pads = list(
                zip(*[
                    calc_same_lower_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        else:
            assert False, "auto_pad should be set if pads is not specified"

    @core.inline
    def f_max_pool_2d(X, Y):
        # yapf: disable
        'nid: L_n'
        for n in range(X.shape(0)):
            'nid: L_c'
            for c in range(X.shape(1)):
                'nid: L_h'
                for h in range(Y.shape(2)):
                    'nid: L_w'
                    for w in range(Y.shape(3)):
                        'nid: init'
                        Y[n, c, h, w] = core.min_value(X.dtype)
                        'nid: L_kh'
                        for kh in range(kernel_shape[0]):
                            'nid: L_kw'
                            for kw in range(kernel_shape[1]):
                                # h_in = h * stride + kh * dilation - pad
                                # w_in = w * stride + kw * dilation - pad
                                if (
                                        h * strides[0] + kh * dilations[0] - pads[0] >= 0 and
                                        h * strides[0] + kh * dilations[0] - pads[0] < X.shape(2) and
                                        w * strides[1] + kw * dilations[1] - pads[1] >= 0 and
                                        w * strides[1] + kw * dilations[1] - pads[1] < X.shape(3)):
                                    'nid: compute'
                                    Y[n, c, h, w] = core.max(
                                        Y[n, c, h, w],
                                        X[n, c,
                                            h * strides[0] + kh * dilations[0] - pads[0],
                                            w * strides[1] + kw * dilations[1] - pads[1]])
        # yapf: enable

    return f_max_pool_2d


def max_pool(io_mem,
             idx_dtype="int32",
             auto_pad: str = 'NOTSET',
             dilations: Optional[Sequence[int]] = None,
             kernel_shape: Sequence[int] = None,
             pads: Optional[Sequence[int]] = None,
             strides: Optional[Sequence[int]] = None):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    # TODO: ceil_mode
    # TODO: return_indices

    if dilations is None:
        dilations = [1 for i in range(n_spatial_dim)]
    if strides is None:
        # NOTE: strides default to 1 in ONNX, while default to kernel_shape in PyTorch
        strides = [1 for i in range(n_spatial_dim)]
    if pads is None:
        if auto_pad == 'VALID':
            pads = list(zip(*[[0, 0] for i in range(n_spatial_dim)]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_UPPER':
            pads = list(
                zip(*[
                    calc_same_upper_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_LOWER':
            pads = list(
                zip(*[
                    calc_same_lower_pad(dilations[i], kernel_shape[i],
                                        strides[i])
                    for i in range(n_spatial_dim)
                ]))
            pads = pads[0] + pads[1]
        else:
            assert False, "auto_pad should be set if pads is not specified"

    @core.inline
    def f_max_pool_2d(X):
        Y_shape = core.create_var((4,), idx_dtype, "output",
                                  io_mem)  # N * C * H * W
        Y_shape[0] = X.shape(0)
        Y_shape[1] = X.shape(1)
        Y_shape[2] = calc_out_size(X.shape(2), dilations[0], kernel_shape[0],
                                   pads[0], pads[2], strides[0])
        Y_shape[3] = calc_out_size(X.shape(3), dilations[1], kernel_shape[1],
                                   pads[1], pads[3], strides[1])
        'nid: V_Y'
        Y = core.create_var(Y_shape, X.dtype, "output", io_mem)
        'nid: recur'
        max_pool_(io_mem, auto_pad, dilations, kernel_shape, pads, strides)(X,
                                                                            Y)
        return Y

    return f_max_pool_2d


def global_avg_pool_(io_mem):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    @core.inline
    def f_global_avg_pool_2d(X, Y):
        'nid: L_n'
        for n in range(X.shape(0)):
            'nid: L_c'
            for c in range(X.shape(1)):
                'nid: init'
                Y[n, c] = 0
                'nid: L_h'
                for h in range(X.shape(2)):
                    'nid: L_w'
                    for w in range(X.shape(3)):
                        'nid: compute'
                        Y[n, c] += X[n, c, h, w]
                'nid: flush'
                Y[n, c] /= X.shape(2) * X.shape(3)

    return f_global_avg_pool_2d


def global_avg_pool(io_mem, idx_dtype="int32"):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    @core.inline
    def f_global_avg_pool_2d(X):
        Y_shape = core.create_var((2,), idx_dtype, "output", io_mem)  # N * C
        Y_shape[0] = X.shape(0)
        Y_shape[1] = X.shape(1)
        Y = core.create_var(Y_shape, X.dtype, "output", io_mem)
        'nid: recur'
        global_avg_pool_(io_mem)(X, Y)
        return Y

    return f_global_avg_pool_2d
