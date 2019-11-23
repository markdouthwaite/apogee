from typing import List, Generator
from collections import OrderedDict

from apogee.inference import JunctionTree
from apogee.factors import FactorSet
from apogee.utils.memo import memoize


class GraphicalModel:
    def __init__(self):
        self.variables = OrderedDict()

    def add(self, variable) -> "GraphicalModel":
        self.variables[variable.name] = variable(graph=self)
        return self

    def remove(self, name: str) -> "GraphicalModel":
        del self.variables[name]
        return self

    def index(self, name: str) -> int:
        for i, key in enumerate(self.variables.keys()):
            if key == name:
                return i

    def name(self, index: int) -> str:
        for i, key in enumerate(self.variables.keys()):
            if i == index:
                return key

        raise IndexError("Index not found")

    def fit(self, frame: "DataFrame") -> None:
        for name, variable in self.variables.items():
            variable.fit(frame[variable.scope].values)

    def iterpredict(self, x: dict = None) -> Generator:
        factors = FactorSet(*self.factors)

        engine = JunctionTree.from_factors(factors)

        if x is not None:
            evidence = []
            for key, value in x.items():
                evidence.append([self.index(key), self[key].states.index(value)])

            engine.update_observations(evidence)

        engine.propagate()
        engine.calibrate()

        for marginal in engine.marginals(*factors.vars):
            response = {}
            name = self.name(marginal.scope[0])
            variable = self.variables[name]
            for i, p in enumerate(marginal.normalise().parameters):
                response.update(**{variable.states[i]: p})
            yield {name: response}

    @memoize
    def predict(self, *args, **kwargs):
        return list(self.iterpredict(*args, **kwargs))

    @property
    def factors(self) -> List:
        return [x.factor for x in self.variables.values()]

    def __getitem__(self, item: str) -> "BaseVariable":
        return self.variables[item]

    def __setitem__(self, key: str, value: "BaseVariable"):
        if value.name == key:
            self.add(value)

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, ",".join(self.variables.keys()))
