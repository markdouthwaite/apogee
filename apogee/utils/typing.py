"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

from typing import Optional, Any, Callable, TypeVar


FactorLike = TypeVar("FactorLike")
FactorSetLike = TypeVar("FactorSetLike")
VariableLike = TypeVar("VariableLike")


def castarg(
    pos: Optional[int] = None,
    name: Optional[str] = None,
    argtype: Optional[type] = None,
) -> Callable[[Any], Any]:
    """Wrap a function to cast positional or keyword arguments to the given type."""

    def _wrapper(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def _inner(*args: Optional[Any], **kwargs: Optional[Any]) -> Any:
            if name is not None and pos is not None:
                raise ValueError(
                    "castarg decorator expects either name or pos, not both."
                )
            if pos is not None:
                args = tuple(
                    x if i != pos and x is not None else argtype(x)
                    for i, x in enumerate(args)
                )
            if name is not None and name in kwargs:
                if kwargs[name] is not None:
                    kwargs[name] = argtype(kwargs[name])
            return func(*args, **kwargs)

        return _inner

    return _wrapper
