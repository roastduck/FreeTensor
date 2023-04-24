__all__ = ['EnableAttachBackward']

from .param_ret_dict import ParamRetDict


class EnableAttachBackward:
    '''
    Get backward object (Func, Driver, etc) and other meta data from a forward object

    This class is a Mixin Class. It should be inherited BEFORE other base classes in
    multiple inheritance.
    '''

    def __init__(self, *args, **kvs):
        '''
        Forward all arguments to other base classes

        In Python, super().__init__ calls the next base class in the full inheritance
        graph of the final class, not only base classes of BackwardAttachedMixin.
        See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance
        '''

        super().__init__(*args, **kvs)
        self._backward = None
        self._input_name_to_gradient_name = None
        self._output_name_to_gradient_name = None

    def attach_backward(self, backward,
                        input_name_to_gradient_name: ParamRetDict,
                        output_name_to_gradient_name: ParamRetDict):
        self._backward = backward
        self._input_name_to_gradient_name = input_name_to_gradient_name
        self._output_name_to_gradient_name = output_name_to_gradient_name

    def has_backward(self):
        return self.backward is not None

    @property
    def backward(self):
        return self._backward

    @property
    def input_name_to_gradient_name(self) -> ParamRetDict:
        return self._input_name_to_gradient_name

    @property
    def output_name_to_gradient_name(self) -> ParamRetDict:
        return self._output_name_to_gradient_name
