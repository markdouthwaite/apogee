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

import numpy as np


def find_min_neighbours(matrix: np.ndarray) -> np.ndarray:
    """Find the vertex with the minimum number of neighbouring vertices."""

    scores = np.ones(matrix.shape[0])
    for i in range(matrix.shape[0]):
        s = matrix[:, i].sum()
        scores[i] = s if s > 0 else np.inf

    return np.argmin(scores)


def eliminate_variable(idx: int, matrix: np.ndarray) -> np.ndarray:
    """Eliminate a vertex from a graph in matrix-form."""

    scope = _node_scope(idx, matrix)
    for i in scope:
        for j in scope:
            if i != j:
                matrix[i, j] = 1
                matrix[j, i] = 1

    matrix[idx, :] = 0.0
    matrix[:, idx] = 0.0
    return matrix


def _node_scope(idx: int, matrix: np.ndarray) -> np.ndarray:
    """Find the scope of given vertex."""

    return np.where(matrix[:, idx] > 0)[0]


def get_elimination_ordering(
    matrix: np.ndarray, heuristic: callable = find_min_neighbours
):
    """Compute the elimination ordering of a given graph in matrix-from."""

    count = 0
    ordering = []
    scopes = []
    while count < matrix.shape[0] - 1:
        j = heuristic(matrix)
        scopes.append(_node_scope(j, matrix))
        matrix = eliminate_variable(j, matrix)
        ordering.append(j)
        count += 1

    ordering.append([x for x in range(matrix.shape[0]) if x not in ordering][0])

    return ordering, scopes
