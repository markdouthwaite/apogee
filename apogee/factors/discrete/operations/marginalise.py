import apogee as ap
import numpy as np


def factor_marginalise(a, v):
    """
    Marginalise out the variable 'v' from factor 'a'.

    Parameters
    ----------
    a: Factor-like
        The target factor-like object.
    v: int/any
        The identifier of the variable to be marginalised out.

    Returns
    -------
    scope: ndarray
        An array containing the scope of the resulting factor.
    card: ndarray
        An array containing the cardinality of the resulting factor.
    vals: ndarray
        An array containing the probability distribution of the resulting factor.

    Examples
    --------
    >>> a = Factor([0], [2], [0.1, 0.9])
    >>> b = Factor([1, 0], [2, 2], [[0.2, 0.8], [0.7, 0.3]])
    >>> c = Factor(*factor_marginalise(a, 0))  # generate new factor from factors a and b
        {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1: 0.75}}

    """

    scope = a.scope[
        np.where(a.scope != v)
    ]  # extract remaining variables from scope scope.
    f_map = ap.index_map_1d(
        a.scope, scope
    )  # create a new map of old scope given new scope.
    card = a.cards[f_map]  # extract cardinality of remaining variables

    assignments = ap.cartesian_product(
        *[np.arange(x) for x in card]
    )  # compute assignments only once.
    f_idx = ap.array_index(
        a.assignments[:, f_map], assignments
    )  # extract indices to be combined.

    values = np.zeros(len(assignments), dtype=np.float64)

    avals = a.parameters
    for i in range(len(a.assignments)):
        if values[f_idx[i]] is None:
            values[f_idx[i]] = avals[i]
        else:
            values[f_idx[i]] += avals[i]

    return scope, card, values
