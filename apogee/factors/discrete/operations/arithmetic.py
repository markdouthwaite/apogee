import numpy as np
import apogee as ap
from typing import Tuple, Callable


def factor_arithmetic(a: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                      b: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                      op: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    scope = ap.union(a[0], b[0])  # calculate the new scope.
    maps_a = ap.array_mapping(scope, a[0])  # generate map of scope of a in new scope.
    maps_b = ap.array_mapping(scope, b[0])  # repeat

    card = np.zeros_like(scope, dtype=np.int32)
    card[maps_a] = a[1]
    card[maps_b] = b[1]

    assignments = ap.cartesian_product(*[np.arange(n, dtype=np.int32) for n in card])

    vals = np.empty(len(assignments), dtype=type(a[2][0]))
    a_idx = ap.array_index(assignments[:, maps_a], a[3])
    b_idx = ap.array_index(assignments[:, maps_b], b[3])

    a_vals, b_vals = a[2], b[2]

    for i, (j, k) in enumerate(zip(a_idx, b_idx)):
        vals[i] = op(a_vals[j], b_vals[k])

    return scope, card, vals

"""
    scope = ap.union(a.scope, b.scope)  # calculate the new scope.
    maps_a = ap.maps(scope, a.scope)  # generate map of scope of a in new scope.
    maps_b = ap.maps(scope, b.scope)  # repeat

    card = np.zeros_like(scope, dtype=np.int64)
    card[maps_a] = a.cards
    card[maps_b] = b.cards

    assignments = ap.cartesian_product(*[np.arange(n) for n in card])

    vals = np.empty(len(assignments), dtype=type(a.parameters[0]))
    a_idx = ap.indices(assignments[:, maps_a], a.assignments)
    b_idx = ap.indices(assignments[:, maps_b], b.assignments)
    avals, bvals = a.parameters, b.parameters

    for i, (j, k) in enumerate(zip(a_idx, b_idx)):
        vals[i] = avals[j] * bvals[k]

    return scope, card, vals
"""
