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


def random_factor(
    max_card=3, max_scope=3, max_factor=10, min_factor=1, ftype=lambda x: x
):
    n = np.random.randint(1, max_scope + 1)
    scope = apogee.legacy.random.urandint(min_factor, max_factor, size=(n,))
    card = np.random.randint(2, max_card + 1, n)
    return ftype(scope, card, np.ones(np.product(card)))


def random_factor_graph(n, gtype=lambda x: x, **kwargs):
    factors = []
    for i in range(n):
        factors.append(random_factor(**kwargs))
    return gtype(*factors)
