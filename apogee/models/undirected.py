"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

import json
from typing import List, Generator, Optional, Any, Text
from collections import OrderedDict

from functools import lru_cache

from pandas import DataFrame
from networkx import Graph

from apogee import io
from apogee.inference import JunctionTree
from apogee.factors import FactorSet
from apogee.models.variables import DiscreteVariable

from apogee.utils.typing import castarg, VariableLike, FactorLike


class UndirectedModel:
    """
    An interface for building undirected graphical models.
    """

    _var_types = {
        "discrete": DiscreteVariable
    }

    def __init__(self):
        """Create a new GraphicalModel instance."""
        self._graph = Graph()
        self.variables: OrderedDict = OrderedDict()

    def add(self, variable: VariableLike) -> "GraphicalModel":
        """Add a variable to the model."""

        self._graph.add_node(variable.name, variable=variable)
        self.variables[variable.name] = variable
        for other in variable.neighbours:
            self._graph.add_edge(other, variable.name)

        return self

    def remove(self, name: str) -> "GraphicalModel":
        """Remove a variable from the model."""

        del self.variables[name]

        return self

    def index(self, name: str) -> int:
        """Get the index of the variable with the given name."""

        # this is slow.
        for i, key in enumerate(self.variables.keys()):
            if key == name:
                return i

        raise KeyError(f"Name '{name}' not found.")

    def name(self, index: int) -> str:
        """Get the name of the variable at the given index."""

        # this is slow.
        for i, key in enumerate(self.variables.keys()):
            if i == index:
                return key

        raise IndexError(f"Index '{index}' not found.")

    def fit(self, df: DataFrame) -> "GraphicalModel":
        """
        Fit the model to the provided frame.

        Parameters
        ----------
        df: DataFrame
            A dataframe containing training data. Each column in the frame should
            correspond to a variable in the model. Currently only support categorical
            variables.

        Returns
        -------
        out: GraphicalModel
            The fitted model.

        Notes
        -----
        * Structure learning is not currently supported.

        """

        for name, variable in self.variables.items():
            variable.fit(df[variable.scope].values)

        return self

    def iter_predict(
        self, x: tuple = None, marginals: tuple = None
    ) -> Generator[dict, None, None]:
        """
        Yields marginals for variables in the model.

        Parameters
        ----------
        x: tuple, optional
            A tuple where each element is an iterable with two elements, the first
            element corresponding to the name of a variable for which you have evidence,
            and the second corresponding to the value of that evidence. For example:
                [("rain", "true")]
            If not provided, no evidence will be injected.
        marginals: tuple, optional
            A tuple containing the names of the variables you wish to obtain marginals
            for. By default, marginals for all variables will be returned.

        Yields
        ------
        out: dict
            A dictionary with one key and one value, where the key corresponds to the
            name of a variable, and the value corresponds to a mapping of state names
            to the marginal probability of that state.

        """

        factors = FactorSet(*self.factors)

        engine = JunctionTree.from_factors(factors)

        if x is not None:
            evidence = []
            for key, value in x:
                evidence.append([self.index(key), self[key].states.index(value)])

            engine.update_observations(evidence)

        engine.propagate()
        engine.calibrate()

        if marginals is not None:
            v = [v for v in factors.vars if self.name(v) in marginals]
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
    @castarg(name="marginals", argtype=tuple)
    @lru_cache(256)
    def predict(self, *args: Optional[Any], **kwargs: Optional[Any]) -> dict:
        """Generate predictions."""

        data = {}
        for e in self.iter_predict(*args, **kwargs):
            data.update(**e)

        return data

    @property
    def factors(self) -> List[FactorLike]:
        """Get a list of the factors in the model."""

        return [x.factor for x in self.variables.values()]

    @classmethod
    def from_dict(cls, data: dict):

        model = cls()

        for key, value in data.items():
            if "type" in value:
                flavour = cls._var_types[value["type"]]
                del value["type"]
            else:
                flavour = DiscreteVariable

            model.add(flavour(key, graph=model, **value))

        return model

    @classmethod
    def from_hugin(cls, filename: Text) -> "GraphicalModel":
        data: dict = io.hugin.load(open(filename))
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filename: Text) -> "GraphicalModel":
        data: dict = json.load(open(filename))
        return cls.from_dict(data)

    def __getitem__(self, item: str) -> VariableLike:
        """Return variable with label 'item'."""

        return self.variables[item]

    def __setitem__(self, key: str, value: VariableLike) -> None:
        """Update variable with label 'item' to 'value'."""

        if value.name == key:
            self.add(value)

    def __repr__(self) -> str:
        return "{0}({1})".format(type(self).__name__, ",".join(self.variables.keys()))
