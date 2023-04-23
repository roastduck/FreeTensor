import abc
import copy
import inspect
from typing import Tuple, Callable, Mapping, Any


class JITMeta(type):

    def __init__(self, *args, **kvs):
        super(JITMeta, self).__init__(*args, **kvs)
        self.inner_type = None

    def __getitem__(self, inner_type):
        if self.inner_type is not None:
            raise TypeError("Cannot call __getitem__ on JIT twice")
        new = copy.copy(self)
        new.inner_type = inner_type
        return new


class JIT(metaclass=JITMeta):
    '''
    Declare a function parameter as a JIT parameter

    A function with one or more JIT parameters will be compiled to a JIT template. It can
    be instantiate after the JIT paraemters are provided

    Usage: `x: JIT` or `x: JIT[AnyPythonType]`. The latter form has no syntactic meanings,
    and is only for documentation.

    NOTE 1: The `JIT` type annotation can only be used for parameter of the outer-most
    function intended for `@transform` (or `@optimize`, etc). It can NOT be used for inner
    functions intended for `@inline`.

    NOTE 2: The `JIT` type annoation can only annotated on parameters inside the function
    signature. It is NOT supported in annotations for statements.
    '''

    pass


class JITTemplate(abc.ABC):
    '''
    A template that can be instantiated given concrete arguments

    By calling `instantiate` with actual arguments you are expecting to run a JIT function with,
    an instantiated object will be returned. Subclasses of `JITTemplate` should override
    `instantiate_by_only_jit_args`, and define what is actually returned.

    Parameters
    ----------
    params : OrderedDict
        Parameter list from inspect.signature(func).parameters
    jit_param_names : Sequence
        Sequence of names of JIT parameters in the original order defined in the function
    '''

    def __init__(self, params, jit_param_names):
        self.params = params
        self.jit_param_names = jit_param_names

    @abc.abstractmethod
    def instantiate_by_only_jit_args(self, *jit_args: Tuple[Any, ...]):
        '''
        Get an instance with only JIT arguments

        This function accpets a tuple of arguments. Keyword arguments is NOT supported, so
        memoization can be easier, with considering the order of the arguments.
        '''

        raise NotImplementedError()

    def instantiate(self, *args, **kvs):
        '''
        Get an instance with the arguments you are expecting to run a JIT function with
        '''

        non_jit_args, jit_args = self.separate_args(*args, **kvs)
        if len(jit_args) < len(self.jit_param_names):
            raise TypeError("No enough JIT parameter provided")
        assert len(jit_args) == len(self.jit_param_names)
        return self.instantiate_by_only_jit_args(*jit_args)

    def instantiate_and_call(self, *args, **kvs):
        '''
        Get an instance and call it with the arguments you are expecting to run a JIT function with
        '''

        non_jit_args, jit_args = self.separate_args(*args, **kvs)
        if len(jit_args) < len(self.jit_param_names):
            raise TypeError("No enough JIT parameter provided")
        assert len(jit_args) == len(self.jit_param_names)
        return self.instantiate_by_only_jit_args(*jit_args)(*non_jit_args)

    def separate_args(self, *args, **kvs):
        '''
        Return a list of non-JIT args, and a list of JIT args
        '''

        non_jit_args = []
        jit_args = []
        for i, name in enumerate(self.params):
            is_jit = self.params[name].annotation is JIT
            if i < len(args):
                if name in kvs:
                    raise TypeError(
                        f"got multiple values for argument '{name}'")
                (jit_args if is_jit else non_jit_args).append(args[i])
            elif name in kvs:
                (jit_args if is_jit else non_jit_args).append(kvs[name])
        return non_jit_args, jit_args
