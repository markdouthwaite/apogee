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

import itertools
from functools import reduce

import numpy as np
from numpy import ndarray


def sort(arr: ndarray, reverse: bool = True, **kwargs: dict) -> ndarray:
    """Sort a numpy array, optionally in reverse order."""

    output = np.sort(arr, **kwargs)

    if reverse:
        output = output[::-1]

    return output


def union1d(*args: ndarray) -> ndarray:
    """Compute the union of an arbitrary set of 1d arrays."""

    return reduce(np.union1d, args)


def difference1d(*args: ndarray) -> ndarray:
    """Compute the difference of an arbitrary set of 1d arrays."""

    return reduce(np.setdiff1d, args)


def intersect1d(*args: ndarray) -> ndarray:
    """Compute the intersection of an arbitrary set of 1d arrays."""

    return reduce(np.intersect1d, args)


def equals(a: ndarray, b: ndarray) -> bool:
    """Check if two arrays are exactly equal."""

    if np.all(np.array(a) == np.array(b)):
        return True
    else:
        return False


def contains(*args: ndarray) -> bool:
    """Check if an arbitrary set of arrays are a subset of each other."""

    if len(intersect1d(*args)) > 0:
        return True
    else:
        return False


def subset(a: ndarray, b: ndarray) -> bool:
    """Check if an array is a complete subset of another."""

    if len(a) == len(intersect1d(a, b)):
        return True
    else:
        return False


def array_map(*args: tuple, **kwargs: dict) -> ndarray:
    """Execute a map operation over a tuple of arrays array."""

    return np.asarray(list(map(*args, **kwargs)))


def array_reduce(*args: tuple, **kwargs: dict) -> ndarray:
    """Execute a reduce operation over an array."""

    return np.asarray(list(reduce(*args, **kwargs)))


def array_index(a: ndarray, b: ndarray) -> ndarray:
    """Find the index of the first occurence of elements in 'a' in array 'b'."""

    a = np.asarray(a).tolist()
    b = np.asarray(b).tolist()

    output = []
    for x in a:
        output.append(b.index(x))

    return np.asarray(output, dtype=np.int32)


def array_mapping(a, b):
    """Find a mapping of array a into array b."""

    return np.where(np.asarray(b)[:, None] == np.asarray(a)[None, :])[1]


def index_map(a: np.ndarray, b: np.ndarray):
    """Find a mapping of array a into array b where a and b are arrays."""

    return np.where(a[:, None] == b[None, :])[1]


def index_map_1d(a: np.ndarray, b: np.ndarray, **kwargs):
    """Find a mapping of array a into array b where a and b are 1d arrays."""

    return np.where(np.in1d(a, b, **kwargs))[0]


def ndarange(*args, shape: tuple = None, **kwargs):
    """Generate arange arrays of arbitrary dimensions."""

    arr = np.array([np.arange(*args[i], **kwargs) for i in range(len(args))])
    return arr.reshape(shape) if shape is not None else arr.T


def cartesian_product(*arr) -> np.ndarray:
    """Find a mapping of array a into array b."""

    return np.asarray(list(itertools.product(*arr)))
