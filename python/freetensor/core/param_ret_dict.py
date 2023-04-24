__all__ = ['Parameter', 'Return', 'ParamRetDict']

from typing import Optional


class Parameter:
    '''
    Alias of a parameter of a function by position instead of by name

    `Parameter(n)` represents the n-th parameter (counted from 0)

    `Parameter()` can be used if there is only one parameter
    '''

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def get_name(self, func):
        if len(func.params) == 0:
            raise KeyError(f"{func.name} has no parameter")
        if self.n is not None:
            return func.params[self.n].name
        else:
            if len(func.params) != 1:
                raise KeyError(
                    f"{func.name} has more than one return value, and you"
                    f" need to specify the number of a return value")
            return func.params[0].name

    def __str__(self):
        return f"Parameter({self.n})"


class Return:
    '''
    Alias of a return value of a function by position instead of by name

    `Return(n)` represents the n-th return value (counted from 0)

    `Return()` can be used if there is only one return value
    '''

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def get_name(self, func):
        if len(func.returns) == 0:
            raise KeyError(f"{func.name} has no return value")
        if self.n is not None:
            return func.returns[self.n].name
        else:
            if len(func.returns) != 1:
                raise KeyError(
                    f"{func.name} has more than one return value, and you"
                    f" need to specify the number of a return value")
            return func.returns[0].name

    def __str__(self):
        return f"Return({self.n})"


class ParamRetDict:
    ''' Look an object using either a function parameter or return value's name or position '''

    def __init__(self, func, d):
        self.func = func
        self.d = d

    def __getitem__(self, key):
        try:
            if type(key) is Parameter:
                key = key.get_name(self.func)
            elif type(key) is Return:
                key = key.get_name(self.func)
            return self.d[key]
        except KeyError as e:
            raise KeyError(
                f"There is no {key} in arguments or return values") from e

    def __contains__(self, key):
        # Python's auto fallback from __getitem__ to __contains__ only works for
        # integer index
        try:
            if type(key) is Parameter:
                key = key.get_name(self.func)
            elif type(key) is Return:
                key = key.get_name(self.func)
            return key in self.d
        except KeyError as e:
            return False

    def __str__(self):
        return str(self.d)

    def __iter__(self):
        return iter(self.d)
