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

import apogee.core as ap
from .utils import index_to_assignment, assignment_to_index


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
