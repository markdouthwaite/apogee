cimport cython

import numpy as np
cimport numpy as np

import apogee.core.fast as ap

from apogee.core.arrays import array_index


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef factor_arithmetic(a, b, op):
    """
    Perform some factor arithmetic.
    """

    # This isn't doing all that much to speed things up at this time.
    cdef scope = ap.union1d(a[0], b[0])  # calculate the new scope.
    cdef maps_a = ap.array_mapping(scope, a[0])  # gen. map of scope of a in new scope.
    cdef maps_b = ap.array_mapping(scope, b[0])  # repeat

    cdef card = np.zeros_like(scope, dtype=np.int32)
    card[maps_a] = a[1]
    card[maps_b] = b[1]

    assignments = ap.cartesian_product([np.arange(n, dtype=np.int32) for n in card])

    cdef vals = np.empty(len(assignments), dtype=type(a[2][0]))
    cdef a_idx = array_index(assignments[:, maps_a], a[3])
    cdef b_idx = array_index(assignments[:, maps_b], b[3])

    cdef a_vals = a[2]
    cdef b_vals = b[2]

    for i, (j, k) in enumerate(zip(a_idx, b_idx)):
        vals[i] = op(a_vals[j], b_vals[k])

    return scope, card, vals
