
import operator
import itertools
import functools
from typing import Iterable

def zip_repeat(data):
    max_length = max(len(sublist) for sublist in data)
    return [[sublist[idx % len(sublist)] for sublist in data] for idx in range(max_length)]


def foldl(func, acc, xs):
  return functools.reduce(func, xs, acc)

def flip(func):
    @functools.wraps(func)
    def newfunc(x, y):
        return func(y, x)
    return newfunc

def foldr(func, acc, xs):
    return functools.reduce(flip(func), reversed(xs), acc)

def var(*args):
    if len(args) == 1:
        if isinstance(args[0], Iterable) and not isinstance(args[0], str):
            # force generator evaluation
            return list(args[0])
        return args[0]
    return [*args]

def prefix_each(text, iterable):
    for element in iterable:
        yield f"{text}{element}"

def suffix_each(text, iterable):
    for element in iterable:
        yield f"{element}{text}"

def make_list(*args):
    return [*args]

def conditional(condition, true_branch, false_branch):
    return true_branch if condition else false_branch


BUILTINS = {
# builtins
    'any': any,
    'all': all,
    'chr': chr,
    'ord': ord,
    'divmod': divmod,
    'enumerate': enumerate,
    'len': len,
    'min': min,
    'max': max,
    'map': map,
    'zip': zip,
    'abs': abs,
    'pow': pow,
    'reversed': reversed,
    'range': range,
# operator
    'lt': operator.lt,
    'le': operator.le,
    'eq': operator.eq,
    'ne': operator.ne,
    'ge': operator.ge,
    'gt': operator.gt,
    'not': operator.not_,
    'is': operator.is_,
    'is_not': operator.is_not,
    'add': operator.add,
    'and': operator.and_,
    'floordiv': operator.floordiv,
    'invert': operator.invert,
    'lshift': operator.lshift,
    'mod': operator.mod,
    'mul': operator.mul,
    'matmul': operator.matmul,
    'neg': operator.neg,
    'or': operator.or_,
    'pos': operator.pos,
    'rshift': operator.rshift,
    'sub': operator.sub,
    'truediv': operator.truediv,
    'xor': operator.xor,
    'concat': operator.concat,
    'contains': operator.contains,
    'countOf': operator.countOf,
    'indexOf': operator.indexOf,
# itertools
    'count': itertools.count,
    'cycle': itertools.cycle,
    'repeat': itertools.repeat,
    'accumulate': itertools.accumulate,
    'chain': itertools.chain,
    'compress': itertools.compress,
    'dropwhile': itertools.dropwhile,
    'groupby': itertools.groupby,
    'islice': itertools.islice,
    'pairwise': itertools.pairwise,
    'starmap': itertools.starmap,
    'takewhile': itertools.takewhile,
    'tee': itertools.tee,
    'zip_longest': itertools.zip_longest,
    'product': itertools.product,
    'permutations': itertools.permutations,
    'combinations': itertools.combinations,
    'combinations_with_replacement': itertools.combinations_with_replacement,
# functools
    'bind': functools.partial,
# custom
    'zip_repeat': zip_repeat,
    'foldl': foldl,
    'foldr': foldr,
    'var': var,

    'str': str,
    'int': int,
    'list': make_list,
    'float': float,
    'bool': bool,

    'join': str.join,
    'prefix_each': prefix_each,
    'suffix_each': suffix_each,
    'if': conditional
}
