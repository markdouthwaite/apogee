from typing import List, Generator, Optional, Any
from collections import OrderedDict

from functools import lru_cache

from apogee.inference import JunctionTree
from apogee.factors import FactorSet
from apogee.utils.typing import castarg


class GraphicalModel:
    """
    This class implements an API for building probabilistic graphical models.
    """

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
        # this is slow.
        for i, key in enumerate(self.variables.keys()):
            if i == index:
                return key

        raise IndexError("Index not found.")

    def fit(self, frame: "DataFrame") -> None:
        for name, variable in self.variables.items():
            variable.fit(frame[variable.scope].values)

    def iterpredict(self, x: tuple = None, y: tuple = None) -> Generator:
        factors = FactorSet(*self.factors)

        engine = JunctionTree.from_factors(factors)

        if x is not None:
            evidence = []
            for key, value in x:
                evidence.append([self.index(key), self[key].states.index(value)])

            engine.update_observations(evidence)

        engine.propagate()
        engine.calibrate()

        if y is not None:
            v = [v for v in factors.vars if self.name(v) in y]
        else:
            v = factors.vars

        for marginal in engine.marginals(*v):
            response = {}
            name = self.name(marginal.scope[0])
            variable = self.variables[name]
            for i, p in enumerate(marginal.normalise().parameters):
                response.update(**{variable.states[i]: p})
            yield {name: response}

    @castarg(name="x", argtype=tuple)
    @castarg(name="y", argtype=tuple)
    @lru_cache(256)
    def predict(self, *args: Optional[Any], **kwargs: Optional[Any]):
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
