__all__ = ['Parameter', 'Return', 'ParamRetDict']

from typing import Optional, Sequence


class Parameter:
    '''
    Alias of a parameter of a function by position instead of by name

    `Parameter(n)` represents the n-th parameter (counted from 0)

    `Parameter()` can be used if there is only one parameter
    '''

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def get_name(self, func_name: str, param_names: Sequence[str]):
        param_names = list(param_names)
        if len(param_names) == 0:
            raise KeyError(f"{func_name} has no parameter")
        if self.n is not None:
            return param_names[self.n]
        else:
            if len(param_names) != 1:
                raise KeyError(
                    f"{func_name} has more than one return value, and you"
                    f" need to specify the number of a return value")
            return param_names[0]

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

    def get_name(self, func_name: str, return_names: Sequence[str]):
        return_names = list(return_names)
        if len(return_names) == 0:
            raise KeyError(f"{func_name} has no return value")
        if self.n is not None:
            return return_names[self.n]
        else:
            if len(return_names) != 1:
                raise KeyError(
                    f"{func_name} has more than one return value, and you"
                    f" need to specify the number of a return value")
            return return_names[0]

    def __str__(self):
        return f"Return({self.n})"


class ParamRetDict:
    ''' Look an object using either a function parameter or return value's name or position '''

    def __init__(self,
                 d,
                 *,
                 func=None,
                 func_name: str = None,
                 param_names: Sequence[str] = None,
                 return_names: Sequence[str] = None):
        ''' Either `func` or (`func_name` and `param_names` and `return_names`) should be provided '''

        self.d = d
        if func is None:
            self.func_name = func_name
            self.param_names = param_names
            self.return_names = return_names
        else:
            self.func_name = func.name
            self.param_names = [p.name for p in func.params]
            self.return_names = [r.name for r in func.returns]

    def __getitem__(self, key):
        try:
            if type(key) is Parameter:
                key = key.get_name(self.func_name, self.param_names)
            elif type(key) is Return:
                key = key.get_name(self.func_name, self.return_names)
            return self.d[key]
        except KeyError as e:
            raise KeyError(
                f"There is no {key} in arguments or return values") from e

    def __contains__(self, key):
        # Python's auto fallback from __getitem__ to __contains__ only works for
        # integer index
        try:
            if type(key) is Parameter:
                key = key.get_name(self.func_name, self.param_names)
            elif type(key) is Return:
                key = key.get_name(self.func_name, self.return_names)
            return key in self.d
        except KeyError as e:
            return False

    def __str__(self):
        return str(self.d)

    def __iter__(self):
        return iter(self.d)
