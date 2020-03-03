"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
