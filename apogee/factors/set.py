import numpy as np
from functools import reduce


class FactorSet(object):
    def __init__(self, *factors):
        self.factors = list(factors)

    def add(self, *factors):
        self.factors.extend(list(factors))

    def get(self, *var):
        return [factor for factor in self if np.all(np.isin(var, factor.scope))]

    def remove(self, *factors, inplace=True):
        factors = [factor for factor in self if factor not in factors]
        if inplace:
            self.factors = factors
            return self
        else:
            return FactorSet(*factors)

    def contains(self, *var):
        return all([x in self for x in var])

    @property
    def vars(self):
        return reduce(np.union1d, [factor.scope for factor in self])

    @property
    def cards(self):
        cards = []
        for var in self.vars:
            for factor in self:
                if var in factor.scope:
                    cards.append(factor.card(var))
                    break
        return np.asarray(cards)

    def blanket(self, *var):
        return FactorSet(*self.get(*var))

    def product(self, **kwargs):
        factor = self.factors[0]
        for other in self.factors[1:]:
            factor *= other
        return factor

    def maximise(self, **kwargs):
        return self.product(**kwargs).maximise()

    def reduce(self, *args, **kwargs):
        return FactorSet(*[x.reduce(*args, **kwargs) for x in self])

    def normalise(self, inplace=False, **kwargs):
        factors = [factor.normalise(inplace=inplace, **kwargs) for factor in self.factors]
        if inplace:
            return self
        else:
            return FactorSet(*factors)

    def apply(self, attrib, *args, **kwargs):
        return [getattr(factor, attrib)(*args, **kwargs) for factor in self]

    def where(self, scope, exact=True):
        if exact:
            return FactorSet(*[x for x in self if np.all(x.scope == scope)])
        else:
            return FactorSet(*[x for x in self if np.all(np.isin(scope, x.scope))])

    def new_factor(self, scope):
        if all(True if x in self.vars else False for x in scope):

            complete_subset = np.asarray(self.where(scope, exact=False).factors)
            if len(complete_subset) > 0:
                factor = complete_subset[np.argmin([x.scope.shape[0] for x in complete_subset])]
                return factor.subset(scope)
            else:
                factors = []
                for var in scope:
                    partial_subset = np.asarray(self.where([var], exact=False).factors)
                    factors.append(partial_subset[np.argmin([x.scope.shape[0] for x in partial_subset])].copy())

                factors = [factor.marginalise(*np.setdiff1d(factor.scope, scope)) for factor in factors]
                factors = FactorSet(*factors)
                return factors.product().subset(scope)

        else:
            missing = ", ".join([str(x) for x in scope if x not in self.vars])
            raise ValueError("Cannot create a new factor as the following variables are not in the set: {0}.".format(missing))

    @property
    def adjacency_matrix(self):
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
        return "{0}({1})".format(type(self).__name__, ", ".join([str(factor) for factor in self]))
