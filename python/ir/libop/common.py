class StaticType:
    ''' The static part of a tensor data type '''

    def __init__(self, elem_type, ndim: int):
        self.elem_type = elem_type
        self.ndim = ndim

    def one_less_dim(self):
        return StaticType(self.elem_type, self.ndim - 1)
