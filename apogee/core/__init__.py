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

from .arrays import (
    index_map_1d,
    index_map,
    cartesian_product,
    intersect1d,
    contains,
    ndarange,
    difference1d,
    union1d,
    array_mapping,
    array_index,
)
from .entropy import (
    entropy,
    cross_entropy,
    symmetric_relative_entropy,
    symmetric_kullback_leibler_divergence,
    kullback_leibler_divergence,
    relative_entropy,
)
from .scaling import normalise
from .search import get_elimination_ordering, find_min_neighbours

__all__ = [
    "normalise",
    "get_elimination_ordering",
    "find_min_neighbours",
    "entropy",
    "cross_entropy",
    "symmetric_relative_entropy",
    "symmetric_kullback_leibler_divergence",
    "kullback_leibler_divergence",
    "relative_entropy",
    "cartesian_product",
    "contains",
    "index_map_1d",
    "index_map",
    "intersect1d",
    "array_index",
    "array_mapping",
    "ndarange",
    "union1d",
    "difference1d",
]
