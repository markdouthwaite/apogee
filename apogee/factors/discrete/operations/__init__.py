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

from .divide import factor_division
from .marginalise import factor_marginalise
from .maximise import factor_maximise
from .product import factor_product
from .random import random_factor, random_factor_graph
from .reduce import factor_reduce
from .subtract import factor_difference
from .sum import factor_sum
from .utils import (
    index_to_assignment,
    assignment_to_index,
    ones_like_card,
    zeros_like_card,
    get_cardinality,
    format_discrete_marginals,
)

__all__ = [
    "index_to_assignment",
    "assignment_to_index",
    "ones_like_card",
    "zeros_like_card",
    "get_cardinality",
    "format_discrete_marginals",
    "factor_difference",
    "factor_division",
    "factor_marginalise",
    "factor_maximise",
    "factor_product",
    "factor_reduce",
    "factor_sum",
    "random_factor",
    "random_factor_graph",
]
