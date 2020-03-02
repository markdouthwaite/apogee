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

import functools
import itertools

from typing import Optional, Type, Union, List

import numpy as np


def maximum_likelihood_update(
    x: np.ndarray,
    states: np.ndarray,
    n: Optional[int] = 0,
    alpha: Optional[float] = 0.0,
    parameters: Optional[Union[np.ndarray, List]] = None,
    dtype: Optional[Type] = np.float32,
) -> np.ndarray:
    """
    Compute the simple MLE update parameters for a dataset.

    # Todo - This can be accelerated.
    """

    scope_states = np.unique(states, axis=0)

    parameters = (
        np.asarray(parameters)
        if parameters is not None
        else np.zeros(scope_states.shape[0], dtype=dtype)
    )

    parameters *= n

    counts = (
        _compute_rowwise_counts(
            x, np.arange(scope_states.shape[1]), scope_states
        ).astype(dtype)
        + alpha
    )

    for i in range(counts.shape[0]):
        idx = np.all(np.equal(scope_states[i][1:], scope_states[:, 1:]), axis=1)
        parameters[idx] += counts[idx]
        parameters[idx] /= parameters[idx].sum()

    return parameters


def _check_cardinality(data, scope, card):
    if card is None:
        card = count_unique(data[:, scope].T, axis=1)
    assert np.all(count_unique(data[:, scope].T, axis=1) == card)
    return card


def _compute_rowwise_counts(data, idx, states):
    count_matching_rows = functools.partial(count_matching, data[:, idx], axis=1)
    return np.asarray([count_matching_rows(states[i]) for i in range(states.shape[0])])


def _compute_states(card):
    variable_unique_states = np.asarray([np.arange(x) for x in card])
    state_combinations = list(itertools.product(*variable_unique_states))
    return np.asarray(state_combinations)


def count_unique(a, axis=-1):
    b = np.sort(a, axis=axis)
    return (b[:, 1:] != b[:, :-1]).sum(axis) + 1


def count_matching(arr, target, axis=1):
    return np.all(np.equal(arr, target), axis=axis).sum()
