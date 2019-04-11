import numpy as np


def find_min_neighbours(matrix):
    scores = np.ones(matrix.shape[0])
    for i in range(matrix.shape[0]):
        s = matrix[:, i].sum()
        scores[i] = s if s > 0 else np.inf
    return np.argmin(scores)


def eliminate_variable(idx, matrix):
    scope = _node_scope(idx, matrix)
    for i in scope:
        for j in scope:
            if i != j:
                matrix[i, j] = 1
                matrix[j, i] = 1

    matrix[idx, :] = 0.0
    matrix[:, idx] = 0.0
    return matrix


def _node_scope(idx, matrix):
    return np.where(matrix[:, idx] > 0)[0]


def get_elimination_ordering(adjacency_matrix):
    count = 0
    ordering = []
    scopes = []
    while count < adjacency_matrix.shape[0] - 1:
        j = find_min_neighbours(adjacency_matrix)
        scopes.append(_node_scope(j, adjacency_matrix))
        adjacency_matrix = eliminate_variable(j, adjacency_matrix)
        ordering.append(j)
        count += 1

    ordering.append(
        [x for x in range(adjacency_matrix.shape[0]) if x not in ordering][0]
    )
    return ordering, scopes
