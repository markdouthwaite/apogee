import numpy as np

import apogee.core as ap


def factor_reduce(factor, evidence, val=0.0):
    if np.any(np.isin(factor.scope, evidence[0])):
        assignments = factor.assignments
        parameters = np.ones(len(assignments)) * val

        factor_map = ap.index_map_1d(factor.scope, [evidence[0]])
        for i, assignment in enumerate(assignments):
            if assignment[factor_map] == int(evidence[1]):
                parameters[i] = factor.parameters[i]

        return factor.scope, factor.cards, parameters
    else:
        return factor.scope, factor.cards, factor.parameters
