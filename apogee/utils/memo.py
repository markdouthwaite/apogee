from functools import wraps


def memoize(func):
    cache = dict()

    @wraps(func)
    def memoized_func(*args):
        cargs = str(args)
        if cargs in cache:
            return cache[cargs]
        result = func(*args)
        cache[cargs] = result
        return result

    return memoized_func
