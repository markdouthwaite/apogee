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

from functools import reduce

import numpy as np


class FactorSet(object):
    """Class representing a set of Factor objects."""

    def __init__(self, *factors) -> None:
        """Initialise a new FactorSet object."""

        self.factors = list(factors)

    def add(self, *factors) -> None:
        """Add one or more factors to the current set."""

        factors = [factor for factor in factors if factor not in self]
        self.factors.extend(list(factors))

    def get(self, *var) -> list:
        """Get Factors associated with one or more variables."""

        return [factor for factor in self if np.all(np.isin(var, factor.scope))]

    def remove(self, *factors, inplace: bool = True) -> "FactorSet":
        """Remove one or more Factors from the current set."""

        factors = [factor for factor in self if factor not in factors]
        if inplace:
            self.factors = factors
            return self
        else:
            return FactorSet(*factors)

    def contains(self, *var) -> bool:
        """Check if the factor set contains one or more variables."""

        return all([x in self for x in var])

    @property
    def vars(self):
        """Get the union of all variables in all Factors in the FactorSet."""

        return reduce(np.union1d, [factor.scope for factor in self])

    @property
    def cards(self) -> list:
        """Get the union of all cardinalities of all variables in all Factors in the FactorSet."""

        cards = []
        for var in self.vars:
            for factor in self:
                if var in factor.scope:
                    cards.append(factor.card(var))
                    break

        return cards

    def blanket(self, *var) -> "FactorSet":
        """Get the FactorSet associated with a given variable set."""

        return FactorSet(*self.get(*var))

    def product(self) -> "Factor":
        """Compute the Joint Probability Distribution over the set."""

        factor = self.factors[0]
        for other in self.factors[1:]:
            factor *= other

        return factor

    def maximise(self, **kwargs) -> "Factor":
        """Maximise the Joint Probability Distribution over the set."""

        return self.product(**kwargs).maximise()

    def reduce(self, *args, **kwargs) -> "FactorSet":
        """Compute the reduced FactorSet (i.e. account for evidence)."""

        return FactorSet(*[x.reduce(*args, **kwargs) for x in self])

    def normalise(self, inplace: bool = False, **kwargs) -> "FactorSet":
        """Apply a normalisation transformation over the set."""

        factors = [
            factor.normalise(inplace=inplace, **kwargs) for factor in self.factors
        ]
        if inplace:
            return self
        else:
            return FactorSet(*factors)

    def apply(self, attrib: str, *args, **kwargs) -> list:
        """Apply an arbitrary method of a Factor over the set."""

        return [getattr(factor, attrib)(*args, **kwargs) for factor in self]

    def where(self, scope: list, exact: bool = True) -> "FactorSet":
        """Compute the FactorSet where a scope exactly matches a provided value."""

        if exact:
            return FactorSet(*[x for x in self if np.all(x.scope == scope)])
        else:
            return FactorSet(*[x for x in self if np.all(np.isin(scope, x.scope))])

    def new_factor(self, scope: list) -> "Factor":
        """Build a new factor from a provided scope."""

        if all(True if x in self.vars else False for x in scope):
            complete_subset = np.asarray(self.where(scope, exact=False).factors)
            if len(complete_subset) > 0:
                factor = complete_subset[
                    np.argmin([x.scope.shape[0] for x in complete_subset])
                ]
                return factor.subset(scope)
            else:
                factors = []
                for var in scope:
                    partial_subset = np.asarray(self.where([var], exact=False).factors)
                    factors.append(
                        partial_subset[
                            np.argmin([x.scope.shape[0] for x in partial_subset])
                        ].copy()
                    )

                factors = [
                    factor.marginalise(*np.setdiff1d(factor.scope, scope))
                    for factor in factors
                ]
                factors = FactorSet(*factors)
                return factors.product().subset(scope)

        else:
            missing = ", ".join([str(x) for x in scope if x not in self.vars])
            raise ValueError(
                "Cannot create a new factor as the following variables are not in "
                "the set: {0}.".format(missing)
            )

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix of the FactorSet."""

        adj = np.zeros((self.vars.shape[0], self.vars.shape[0]))
        for i, f in enumerate(self.factors):
            for vj in f.scope:
                for vk in f.scope:
                    j, k = vj, vk
                    if j != k:
                        adj[j, k] = 1.0
        return adj

    def __len__(self):
        return len(self.factors)

    def __iter__(self):
        for factor in self.factors:
            yield factor

    def __repr__(self):
        return "{0}({1})".format(
            type(self).__name__, ", ".join([str(factor) for factor in self])
        )
