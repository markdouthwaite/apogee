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

from abc import ABC, abstractmethod
from copy import copy

import numpy as np


class Factor(ABC):
    """Abstract Class for Apogee Factors."""

    def __init__(self, scope):
        """
        A base structure for Factor objects.

        Parameters
        ----------
        scope: array_like
            An array of integer values corresponding to the variables in the scope of
            this Factor.

        References
        ----------
        [1] D. Koller, N. Freidman: Probabilistic Graphical Models, Principles and
            Techniques
        [2] F. Jensen: Bayesian Networks

        """

        self.scope = np.asarray(scope).astype(np.int32)

        assert len(np.unique(self.scope)) == len(self.scope)

    def copy(self):
        """Create a copy of the current Factor."""

        return copy(self)

    @abstractmethod
    def normalise(self, *args, **kwargs):
        """Normalise the factor's distribution."""

        pass

    @abstractmethod
    def product(self, *other):
        """Perform the factor product operation on the Factor and target other Factors."""

        pass

    @abstractmethod
    def maximise(self, *other):
        """
        Perform the factor maximisation (marginalisation) operation on the Factor and target
        other Factors.
        """

        pass

    @abstractmethod
    def marginalise(self, *other):
        """Perform the factor marginalisation operation on the Factor and target other Factors."""

        pass

    @abstractmethod
    def reduce(self, *evidence):
        """Perform the factor reduce operation on the Factor and target other Factors."""

        pass

    @abstractmethod
    def vacuous(self, mapping=None):
        """Generate an 'empty' factor object (like clone, but creates a default distribution)"""

        pass

    @abstractmethod
    def assignment(self, *args):
        """Returns the value the Factor's distribution at the given assignment."""

        pass

    @abstractmethod
    def division(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def entropy(self):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, value):
        pass

    def _operation(self, others, operation, inplace=False, **kwargs):
        """
        Perform a specified operation on the factor.

        Parameters
        ----------
        others: iterable of BaseFactor-like objects.
            The Factors upon which the current Factor is to operate.
        operation: function
            The function that performs the operation to be applied. Must return a Factor object.
        inplace: bool
            Specify whether the operation is to be applied to the current Factor, or returns a new
            Factor (leaving the current Factor untouched).

        Returns
        -------
        out: BaseFactor-like
            The resulting Factor produced by the operation, either a reference to a new Factor, or
            the updated current Factor object.

        """

        factor = self
        for other in tuple(others):  # mildly ugly -> wrap operation outputs?
            factor = type(self)(*operation(factor, other, **kwargs))

        if inplace:
            return self._update(factor)
        else:
            return factor

    @abstractmethod
    def subset(self, scope):
        pass

    def _update(self, factor, *args):
        """
        Update the Factor's parameters to match those in the passed Factor object. (Redo this)

        Parameters
        ----------
        factor: BaseFactor-like
            The Factor for which the current Factor's parameters should be updated to reflect.

        """

        self.scope = factor.scope
        return self

    def __repr__(self):
        return "{0}({1})".format(
            type(self).__name__, ",".join([str(x) for x in self.scope])
        )

    def __mul__(self, other):
        return self.product(other)

    def __add__(self, other):
        return self.sum(other)

    def __sub__(self, other):
        return self.difference(other)

    def __truediv__(self, other):
        return self.division(other)

    def __iter__(self):
        for variable in self.scope:
            yield variable

    def __len__(self):
        return len(self.scope)
