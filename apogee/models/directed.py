"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

from networkx import DiGraph
from .undirected import UndirectedModel
from apogee.models.variables import DiscreteVariable


class DirectedModel(UndirectedModel):
    def __init__(self):
        super().__init__()
        self._graph = DiGraph()

    @classmethod
    def from_dict(cls, data: dict):

        model = cls()

        for key, value in data.items():
            if "type" in value:
                flavour = cls._var_types[value["type"]]
                del value["type"]
            else:
                flavour = DiscreteVariable

            if "parents" in value:
                value["neighbours"] = value["parents"]
                del value["parents"]

            model.add(flavour(key, graph=model, **value))

        return model


BayesianNetwork = DirectedModel
