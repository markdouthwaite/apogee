import numpy as np
import apogee as ap
from .operations import *
from apogee.factors.base import Factor
from .optimise import maximum_likelihood_update
from .operations import (
    factor_product,
    factor_division,
    factor_marginalise,
    factor_maximise,
    factor_reduce,
    factor_sum,
)


class DiscreteFactor(Factor):
    def __init__(
        self, scope, cardinality, parameters=None, alpha=0.0, samples=0, **kwargs
    ):
        """
        A class representing a discrete stochastic factor.

        Parameters
        ----------
        scope: array_like, integer
            An array of integers corresponding to the variables in the scope of the current factor.
            Note that the ordering of this array is important -- make sure the scope mapping is
            correct!
        cardinality: array_like, integer
            An array of integers corresponding to the cardinality of each of the variable in the
            scope of the factor. Once again, note that the order of this array should align exactly
            with the 'scope' array.
        parameters: array_like, float
            An array of floating point numbers representing the distribution of the factor. The
            factor expects to receive these parameters in log-space. Set the transform keyword to
            apply a transform to the parameters.

        References
        ----------
        D. Koller, N. Freidman: Probabilistic Graphical Models, Principles and Techniques
        F. Jensen: Bayesian Networks

        """

        super(DiscreteFactor, self).__init__(scope)
        self._samples = samples
        self._alpha = alpha
        self._cardinality = self._init_cards(cardinality)
        self._parameters = self._init_params(parameters, **kwargs)

    def fit(self, x, y=None):
        return self.fit_partial(x, y)

    def fit_partial(self, x, y=None):
        # TODO: add tests.
        if y is not None:
            x = np.c_[y, x]
        self._parameters = maximum_likelihood_update(
            x, self.assignments, parameters=self.p, alpha=self._alpha, n=self._samples
        )
        self._samples += x.shape[0]
        return self

    def predict(self, x):
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

    def sum(self, *others, **kwargs):
        return self._operation(others, factor_sum, **kwargs)

    def product(self, *others, **kwargs):
        return self._operation(others, factor_product, **kwargs)

    def division(self, *others, **kwargs):
        return self._operation(others, factor_division, **kwargs)

    def normalise(self, inplace=False, row_wise=True, **kwargs):
        if row_wise:
            values = self._row_wise_scaling(**kwargs)
        else:
            values = self._scaling(**kwargs)

        if inplace:
            self._parameters = values
            return self
        else:
            return DiscreteFactor(self.scope, self.cards, values)

    def maximise(self, *others, **kwargs):
        return self._operation(others, factor_maximise, **kwargs)

    def marginalise(self, *others, **kwargs):
        return self._operation(others, factor_marginalise, **kwargs)

    def reduce(self, *evidence, **kwargs):
        return self._operation(evidence, factor_reduce, **kwargs)

    def mpe(self, mode="max", **kwargs):
        if mode == "min":
            return self.assignments[self.argmin(**kwargs)]
        else:
            return self.assignments[self.argmax(**kwargs)]

    def max(self, **kwargs):
        return np.max(self.parameters, **kwargs)

    def min(self, **kwargs):
        return np.min(self.parameters, **kwargs)

    def argmax(self, **kwargs):
        return np.argmax(self.parameters, **kwargs)

    def argmin(self, **kwargs):
        return np.argmin(self.parameters, **kwargs)

    def log(self, inplace=True):
        parameters = np.log(np.clip(self._parameters.copy(), 1e-6))
        if inplace:
            self._parameters = parameters
            return self
        else:
            return DiscreteFactor(self.scope, self.cards, parameters)

    def exp(self, inplace=True):
        parameters = np.exp(self._parameters.copy())
        if inplace:
            self._parameters = parameters
            return self
        else:
            return DiscreteFactor(self.scope, self.cards, parameters)

    def card(self, variable):
        return self.cards[ap.array_mapping(self.scope, [variable])]

    # def value(self, *asn, **kwargs):
    #     return self.parameters[self.partial_index(*asn, **kwargs)]

    def subset(self, scope):
        cards = [self.card(x)[0] for x in scope]
        return DiscreteFactor(scope, cards).identity

    @property
    def entropy(self):
        return ap.entropy(self._parameters)

    def index(self, assignment):
        return assignment_to_index(
            np.atleast_1d(np.asarray(assignment, dtype=np.int64)), self.cards
        )

    def vacuous(self, *args, c=1.0, **kwargs):
        return type(self)(
            self.scope, self.cards, c * np.ones_like(self.parameters), **kwargs
        )

    def assignment(self, index):
        return index_to_assignment(index, self.cards)

    def _init_params(self, params, transform=None, fill=0.0):
        if params is None:
            _params = ones_like_card(self.cards) * fill
        else:
            _params = np.asarray(params, dtype=np.float32)
            m, n = len(_params), np.product(self.cards)
            assert m == n

        return _params if transform is None else transform(_params)

    @staticmethod
    def _init_cards(cards):
        _cards = np.asarray(cards, dtype=np.int32)
        assert np.all([x >= 1 for x in cards])
        return _cards

    def _scaling(self, epsilon=1e-16, **kwargs):
        # TODO: spin out
        return ap.normalise(self.parameters.copy(), a_min=epsilon, **kwargs)

    def _row_wise_scaling(self, epsilon=1e-16):
        # TODO: spin out
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

    def _update(self, factor, *args):
        """
        Update the Factor's parameters to match those in the passed Factor object. (Redo this)

        Parameters
        ----------
        factor: BaseFactor-like
            The Factor for which the current Factor's parameters should be updated to reflect.

        """

        self.scope = factor.scope
        self.cards = factor.cards
        self.parameters = factor.parameters
        return self

    @property
    def k(self):
        return len(self.cards)

    @property
    def n(self):
        return len(self.parameters)

    @property
    def p(self):
        return self.parameters

    @property
    def cards(self):
        return self._cardinality

    @cards.setter
    def cards(self, value):
        self._init_cards(value)

    @property
    def assignments(self):
        return ap.cartesian_product(*[np.arange(n) for n in self.cards])

    @property
    def identity(self):
        return self.vacuous()

    @property
    def marginals(self):
        return [self.marginalise(*ap.difference(self.scope, [v])) for v in self.scope]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = self._init_params(value)
