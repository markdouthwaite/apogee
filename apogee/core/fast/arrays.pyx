"""Auxiliary array operations."""

import itertools
from functools import reduce

import numpy as np
cimport numpy as np


cpdef union1d(a, b):
    """Compute the union of an arbitrary set of 1d arrays."""

    return reduce(np.union1d, (a, b))


cpdef difference1d(a, b):
    """Compute the difference of an arbitrary set of 1d arrays."""

    return reduce(np.setdiff1d, (a, b))


cpdef intersect1d(a, b):
    """Compute the intersection of an arbitrary set of 1d arrays."""

    return reduce(np.intersect1d, (a, b))


cpdef equals(a, b):
    """Check if two arrays are exactly equal."""

    if np.all(np.array(a) == np.array(b)):
        return True
    else:
        return False


cpdef contains(a, b):
    """Check if an arbitrary set of arrays are a subset of each other."""

    if len(intersect1d(a, b)) > 0:
        return True
    else:
        return False


cpdef subset(a, b):
    """Check if an array is a complete subset of another."""

    if len(a) == len(intersect1d(a, b)):
        return True
    else:
        return False


cdef array_index(a, b):
    """Find the index of the first occurence of elements in 'a' in array 'b'."""

    a = np.asarray(a).tolist()
    b = np.asarray(b).tolist()

    output = []
    for x in a:
        output.append(b.index(x))

    return np.asarray(output, dtype=np.int32)


cpdef array_mapping(a, b):
    """Find a mapping of array a into array b."""

    return np.where(np.asarray(b)[:, None] == np.asarray(a)[None, :])[1]


cpdef index_map(a, b):
    """Find a mapping of array a into array b where a and b are arrays."""

    return np.where(a[:, None] == b[None, :])[1]


cpdef index_map_1d(a, b):
    """Find a mapping of array a into array b where a and b are 1d arrays."""

    return np.where(np.in1d(a, b))[0]


cpdef cartesian_product(args):
    """Find a mapping of array a into array b."""

    return np.asarray(list(itertools.product(*args)))
