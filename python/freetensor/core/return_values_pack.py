__all__ = ['ReturnValuesPack']

from typing import Iterable


class ReturnValuesPack:
    '''
    Hold return values from a Driver invocation

    Return values can be retrieved in an anonymous manner: `x, y, z = pack`,
    or in a named manner: `pack['x']`

    Please note that a ReturnValuesPack is different from a OrderedDict, as
    OrderedDict unpacks to keys rather than values
    '''

    def __init__(self, keys: Iterable[str], values: Iterable):
        keys = list(keys)
        values = list(values)
        assert len(keys) == len(values)
        self.keys = keys
        self.values = values

    def __iter__(self):
        ''' Get all return values in the order declared in Func '''
        yield from self.values

    def __getitem__(self, key):
        ''' Get a return value with a name. Tuple is supported for multiple values '''
        if type(key) is tuple or type(key) is list:
            ret = []
            for k in key:
                ret.append(self[k])
            return ret
        if isinstance(key, int) or isinstance(key, slice):
            return self.values[key]
        for k, v in zip(self.keys, self.values):
            if k == key:
                return v
        raise KeyError("No such return value named " + key)

    def __contains__(self, key):
        ''' Test if a return value exists '''
        for k, v in zip(self.keys, self.values):
            if k == key:
                return True
        return False
