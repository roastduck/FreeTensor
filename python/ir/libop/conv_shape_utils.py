def calc_out_size(in_size, dilation, kernel_size, pad_begin, pad_end, stride):
    # Source: https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html
    return (in_size + pad_begin + pad_end - dilation *
            (kernel_size - 1) - 1) // stride + 1


def calc_same_pad(dilation, kernel_size, stride):
    # Solve calc_out_size == ceil(in_size / stride)
    return dilation * (kernel_size - 1) - stride + 1


def calc_same_upper_pad(dilation, kernel_size, stride):
    p = calc_same_pad(dilation, kernel_size, stride)
    return p // 2 + p % 2, p // 2


def calc_same_lower_pad(dilation, kernel_size, stride):
    p = calc_same_pad(dilation, kernel_size, stride)
    return p // 2, p // 2 + p % 2
