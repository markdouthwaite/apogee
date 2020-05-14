import warnings
from typing import Tuple, Callable

import numpy as np

import apogee.core as ap

try:
    from .fast import factor_arithmetic

except ImportError as e:
    print(e)
    warnings.warn("Failed to load accelerated arithmetic operations. Using defaults.")

    def factor_arithmetic(
        a: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        b: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        op: Callable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        scope = ap.union1d(a[0], b[0])  # calculate the new scope.
        maps_a = ap.array_mapping(
            scope, a[0]
        )  # generate map of scope of a in new scope.
        maps_b = ap.array_mapping(scope, b[0])  # repeat

        card = np.zeros_like(scope, dtype=np.int32)
        card[maps_a] = a[1]
        card[maps_b] = b[1]

        assignments = ap.cartesian_product(
            *[np.arange(n, dtype=np.int32) for n in card]
        )

        values = np.empty(len(assignments), dtype=type(a[2][0]))

        a_idx = ap.array_index(assignments[:, maps_a], a[3])
        b_idx = ap.array_index(assignments[:, maps_b], b[3])

        a_values, b_values = a[2], b[2]

        for i, (j, k) in enumerate(zip(a_idx, b_idx)):
            values[i] = op(a_values[j], b_values[k])

        return scope, card, values
