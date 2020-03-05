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

from typing import Optional, Any, Union, List, Type
import numpy as np
from numpy import ndarray
import apogee.core as ap
from apogee.factors.base import Factor
from .operations import *
from .operations import (
    factor_product,
    factor_division,
    factor_marginalise,
    factor_maximise,
    factor_reduce,
    factor_sum,
)
from .optimise import maximum_likelihood_update


class DiscreteFactor(Factor):
    def __init__(
        self,
        scope: ndarray,
        cardinality: ndarray,
        parameters: Optional[ndarray] = None,
        alpha: Optional[float] = 0.0,
        samples: Optional[int] = 0,
        **kwargs: Optional[Any],
    ) -> None:
        """
        A class representing a discrete stochastic factor.

        Parameters
        ----------
        scope: array_like, integer
            An array of integers corresponding to the variables in the scope of the
            current factor. Note that the ordering of this array is important -- make
            sure the scope mapping is
            correct!
        cardinality: array_like, integer
            An array of integers corresponding to the cardinality (number of states) of
            each of the variable in the scope of the factor. Once again, note that the
            order of this array should align exactly with the 'scope' array.
        parameters: array_like, float
            An array of floating point numbers representing the distribution of the
            factor. The factor expects to receive these parameters in log-space. Set
            the transform keyword to apply a transform to the parameters.
        alpha: float
            A prior, currently a fixed value, to be applied when fitting the factor to
            a dataset.

        References
        ----------
        D. Koller, N. Freidman: Probabilistic Graphical Models, Principles and
            Techniques
        F. Jensen: Bayesian Networks

        """

        super(DiscreteFactor, self).__init__(scope)
        self._samples = samples
        self._alpha = alpha
        self._cardinality = self._init_cards(cardinality)
        self._parameters = self._init_params(parameters, **kwargs)

    def fit(self, x: ndarray, y: Optional[Union[ndarray, None]] = None) -> Factor:
        return self.fit_partial(x, y)

    def fit_partial(
        self, x: ndarray, y: Optional[Union[ndarray, None]] = None
    ) -> Factor:

        if y is not None:
            x = np.c_[y, x]

        self._parameters = maximum_likelihood_update(
            x, self.assignments, parameters=self.p, alpha=self._alpha, n=self._samples
        )

        self._samples += x.shape[0]

        return self

    def predict(self, x: ndarray) -> ndarray:
        output = []
        for i, z in enumerate(x):
            evidence = [
                [self.scope[j], z[j - 1]] for j in range(1, self.scope[1:].shape[0] + 1)
            ]
            output.append(
                self.reduce(*evidence, inplace=False)
                .marginalise(*[e[0] for e in evidence])
                .argmax()
            )
        return np.asarray(output)

    def sum(self, *others: Factor, **kwargs: Optional[Any]) -> Factor:
        return self._operation(others, factor_sum, **kwargs)

    def product(self, *others: Factor, **kwargs: Optional[Any]) -> Factor:
        return self._operation(others, factor_product, **kwargs)

    def division(self, *others: Factor, **kwargs: Optional[Any]) -> Factor:
        return self._operation(others, factor_division, **kwargs)

    def normalise(
        self,
        inplace: Optional[bool] = False,
        row_wise: Optional[bool] = True,
        **kwargs: Optional[Any],
    ) -> Factor:
        if row_wise:
            values = self._row_wise_scaling(**kwargs)
        else:
            values = self._scaling(**kwargs)

        if inplace:
            self._parameters = values
            return self
        else:
            return DiscreteFactor(self.scope, self.cards, values)

    def maximise(self, *others: Factor, **kwargs: Optional[Any]) -> Factor:
        return self._operation(others, factor_maximise, **kwargs)

    def marginalise(self, *others: Factor, **kwargs: Optional[Any]) -> Factor:
        return self._operation(others, factor_marginalise, **kwargs)

    def reduce(self, *evidence: Factor, **kwargs: Optional[Any]) -> Factor:
        return self._operation(evidence, factor_reduce, **kwargs)

    def mpe(self, mode: str = "max", **kwargs: Optional[Any]) -> ndarray:
        if mode == "min":
            return self.assignments[self.argmin(**kwargs)]
        else:
            return self.assignments[self.argmax(**kwargs)]

    def max(self, **kwargs: Optional[Any]):
        return np.max(self.parameters, **kwargs)

    def min(self, **kwargs: Optional[Any]) -> float:
        return np.min(self.parameters, **kwargs)

    def argmax(self, **kwargs: Optional[Any]) -> ndarray:
        return np.argmax(self.parameters, **kwargs)

    def argmin(self, **kwargs: Optional[Any]) -> ndarray:
        return np.argmin(self.parameters, **kwargs)

    def log(
        self, inplace: Optional[bool] = True, clip: Optional[float] = 1e-6
    ) -> Factor:
        parameters = np.log(np.clip(self._parameters.copy(), clip))
        if inplace:
            self._parameters = parameters
            return self
        else:
            return DiscreteFactor(self.scope, self.cards, parameters)

    def exp(self, inplace: Optional[bool] = True) -> Factor:

        parameters = np.exp(self._parameters.copy())

        if inplace:
            self._parameters = parameters
            return self

        else:
            return DiscreteFactor(self.scope, self.cards, parameters)

    def card(self, variable: int) -> ndarray:
        return self.cards[ap.array_mapping(self.scope, [variable])]

    def subset(self, scope: ndarray) -> Factor:
        cards = [self.card(x)[0] for x in scope]
        return DiscreteFactor(scope, cards).identity

    @property
    def entropy(self) -> Union[float, ndarray]:
        return ap.entropy(self._parameters)

    def index(self, assignment: Union[int, ndarray, List[int]]) -> ndarray:
        return assignment_to_index(
            np.atleast_1d(np.asarray(assignment, dtype=np.int64)), self.cards
        )

    def vacuous(self, *args, c: Optional[float] = 1.0, **kwargs: Optional[Any]):
        return type(self)(
            self.scope, self.cards, c * np.ones_like(self.parameters), **kwargs
        )

    def assignment(self, index: ndarray) -> ndarray:
        return index_to_assignment(index, self.cards)

    def _init_params(
        self,
        params: Optional[ndarray],
        callback: Optional[callable] = None,
        fill: float = 0.0,
    ):
        if params is None:
            _params = ones_like_card(self.cards) * fill
        else:
            _params = np.asarray(params, dtype=np.float32)
            m, n = len(_params), np.product(self.cards)
            assert m == n

        return _params if callback is None else callback(_params)

    @staticmethod
    def _init_cards(cards: ndarray):
        """Initialise and validate an array of cardinalities."""

        _cards = np.asarray(cards, dtype=np.int32)
        if not np.all([x >= 1 for x in cards]):
            raise ValueError(
                "Invalid variable cardinality found: "
                "all variables must have one or more states in a DiscreteFactor"
            )
        return _cards

    def _scaling(self, epsilon: float = 1e-16, **kwargs):
        """Scale the factor's parameters."""

        return ap.normalise(self.parameters.copy(), a_min=epsilon, **kwargs)

    def _row_wise_scaling(self, epsilon: float = 1e-16):
        """Apply row-wise scaling to the factor's parameters."""

        # Todo: this is slow and ugly, fix it.
        values = self.parameters.copy()
        parent_states = ap.cartesian_product(
            *np.array([np.arange(x) for x in self.cards[1:]])
        )
        for parent_state in parent_states:
            idx = []
            row_sum = epsilon
            for state in np.arange(self.cards[0]):
                idx.append(self.index([state, *parent_state]))
                row_sum += values[idx[-1]]
            values[idx] /= row_sum
        return values

    def _update(self, factor: "DiscreteFactor", *args):
        """
        Update the Factor's parameters to match those in the passed Factor object.

        Parameters
        ----------
        factor: BaseFactor-like
            The Factor for which the current Factor's parameters should be updated to reflect.

        """

        # Todo: this needs a rethink.

        self.scope = factor.scope
        self.cards = factor.cards
        self.parameters = factor.parameters
        return self

    @property
    def k(self):
        """Get the number of variables in the factor."""
        return len(self.scope)

    @property
    def n(self):
        """Get the total number of parameters of the factor."""

        return len(self.parameters)

    @property
    def p(self):
        """Alias for `parameter` attribute."""

        return self.parameters

    @property
    def cards(self):
        """Alias for `cardinality` attribute."""

        # Why is cardinality private?
        return self._cardinality

    @cards.setter
    def cards(self, values: ndarray):
        self._cardinality = self._init_cards(values)

    @property
    def assignments(self):
        """Generate a list of the unique states of the factor."""
        return ap.cartesian_product(*[np.arange(n) for n in self.cards])

    @property
    def identity(self):
        """Generate the identity factor for the current factor."""
        return self.vacuous()

    @property
    def marginals(self):
        """Generate the marginalse for each variable in the factor's scope."""
        return [self.marginalise(*ap.difference1d(self.scope, [v])) for v in self.scope]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = self._init_params(value)
