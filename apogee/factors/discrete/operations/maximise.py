from .utils import index_to_assignment, assignment_to_index
import apogee as ap
import numpy as np


def factor_maximise(a, v):
    scope = a.scope[
        np.where(a.scope != v)
    ]  # extract remaining variables from scope scope.
    f_map = ap.index_map_1d(
        a.scope, scope
    )  # create a new map of old scope given new scope.
    card = a.cards[f_map]  # extract cardinality of remaining variables
    assignments = ap.cartesian_product(*[np.arange(n) for n in card])
    values = np.ones(len(assignments), dtype=np.float64) * -np.inf

    avals = a.params

    for i in range(len(a.params)):

        j = assignment_to_index(index_to_assignment(i, a.cards)[f_map], card)

        values[j] = np.max([values[j], avals[i]])

    return scope, card, values
