def castarg(pos=None, name=None, argtype=None):
    def _wrapper(func):
        def _inner(*args, **kwargs):
            if name is not None and pos is not None:
                raise ValueError(
                    "castarg decorator expects either name or pos, not both."
                )
            if pos is not None:
                args = tuple(x if i != pos else argtype(x) for i, x in enumerate(args))
            if name is not None and name in kwargs:
                kwargs[name] = argtype(kwargs[name])
            return func(*args, **kwargs)

        return _inner

    return _wrapper
