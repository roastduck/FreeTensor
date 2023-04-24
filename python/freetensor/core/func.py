__all__ = ['FuncParam', 'FuncRet', 'Func']

from typing import Sequence

import freetensor_ffi as ffi
from freetensor_ffi import FuncParam, FuncRet

from .enable_attach_backward import EnableAttachBackward
from .frontend import lang_overload


class Func(EnableAttachBackward, ffi.Func):

    def __init__(self,
                 name: str,
                 params: Sequence[FuncParam],
                 returns: Sequence[FuncRet],
                 body: ffi.Stmt,
                 extra_closure={},
                 user_grads=[]):
        super().__init__(name, params, returns, body, extra_closure)
        self.user_grads = user_grads

        # Mimic a Python function
        self.__name__ = name

    def __call__(self, *args, **kvs):
        '''
        Enable invoking a transformed AST in another function being transformed, via
        `inlined_invoke`
        '''

        if lang_overload.in_staging():
            if len(self.returns) == 1:
                names = (self.name,)
            else:
                names = tuple(
                    f"{self.name}.{i}" for i in range(len(self.returns)))
            return lang_overload.register_inlined_invoke(names, self, args, kvs)
        else:
            raise lang_overload.error(
                'Unexpected call on a transformed AST. A transformed AST can only '
                'be called in the following two ways: 1) called with actual data '
                'after `@optimize`, and 2) called from another function to be '
                '`@transform`ed')
