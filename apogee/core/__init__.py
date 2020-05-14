"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
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
