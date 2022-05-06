from typing import Sequence, Optional

from .. import core
from .conv_shape_utils import *


@core.inline
def max_pool_(X,
              Y,
              auto_pad: str = 'NOTSET',
              dilations: Optional[Sequence[int]] = None,
              kernel_shape: Sequence[int] = None,
              pads: Optional[Sequence[int]] = None,
              strides: Optional[Sequence[int]] = None):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    # TODO: ceil_mode
    # TODO: return_indices

    if dilations is None:
        dilations = [1] * n_spatial_dim
    if strides is None:
        # NOTE: strides default to 1 in ONNX, while default to kernel_shape in PyTorch
        strides = [1] * n_spatial_dim
    if pads is None:
        if auto_pad == 'VALID':
            pads = list(zip(*([[0, 0]] * n_spatial_dim)))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_UPPER':
            pads = list(
                zip(*[
                    calc_same_upper_pad(dil, kern, stride) for dil, kern, stride
                    in zip(dilations, kernel_shape, strides)
                ]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_LOWER':
            pads = list(
                zip(*[
                    calc_same_lower_pad(dil, kern, stride) for dil, kern, stride
                    in zip(dilations, kernel_shape, strides)
                ]))
            pads = pads[0] + pads[1]
        else:
            assert False, "auto_pad should be set if pads is not specified"

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


@core.inline
def max_pool(X,
             auto_pad: str = 'NOTSET',
             dilations: Optional[Sequence[int]] = None,
             kernel_shape: Sequence[int] = None,
             pads: Optional[Sequence[int]] = None,
             strides: Optional[Sequence[int]] = None):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    # TODO: ceil_mode
    # TODO: return_indices

    if dilations is None:
        dilations = [1] * n_spatial_dim
    if strides is None:
        # NOTE: strides default to 1 in ONNX, while default to kernel_shape in PyTorch
        strides = [1] * n_spatial_dim
    if pads is None:
        if auto_pad == 'VALID':
            pads = list(zip(*([[0, 0]] * n_spatial_dim)))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_UPPER':
            pads = list(
                zip(*[
                    calc_same_upper_pad(dil, kern, stride) for dil, kern, stride
                    in zip(dilations, kernel_shape, strides)
                ]))
            pads = pads[0] + pads[1]
        elif auto_pad == 'SAME_LOWER':
            pads = list(
                zip(*[
                    calc_same_lower_pad(dil, kern, stride) for dil, kern, stride
                    in zip(dilations, kernel_shape, strides)
                ]))
            pads = pads[0] + pads[1]
        else:
            assert False, "auto_pad should be set if pads is not specified"

    Y = core.empty([
        X.shape(0),
        X.shape(1),
        calc_out_size(X.shape(2), dilations[0], kernel_shape[0], pads[0],
                      pads[2], strides[0]),
        calc_out_size(X.shape(3), dilations[1], kernel_shape[1], pads[1],
                      pads[3], strides[1])
    ], X.dtype, X.mtype)
    'nid: recur'
    max_pool_(X, Y, auto_pad, dilations, kernel_shape, pads, strides)
    return Y


@core.inline
def global_avg_pool_(X, Y):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)
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


@core.inline
def global_avg_pool(X):

    n_spatial_dim = 2  # Currently only 2-D convolution is supported (TODO)

    Y = core.empty([X.shape(0), X.shape(1)], X.dtype, X.mtype)
    'nid: recur'
    global_avg_pool_(X, Y)
    return Y
