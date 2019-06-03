import numpy as np
from ...factors import DiscreteFactor


class Variable(object):
    def __init__(
        self, name: str, states: list, parameters: list = None, parents: list = None
    ):
        self.name = name
        self.states = np.asarray(states, dtype=np.str_)
        self.factor = None
        self.parents = np.asarray(parents or [], dtype=np.str_)
        self.parameters = parameters

    def build_factor(self, network):
        variables = [self, *[network[parent] for parent in self.parents]]
        scope = np.asarray(
            [network.id(variable) for variable in variables], dtype=np.int32
        )
        cards = np.asarray([len(var.states) for var in variables], dtype=np.int32)
        if self.parameters is not None:
            params = np.asarray(self.parameters, dtype=np.float32).flatten("F")
        else:
            params = None
        self.factor = DiscreteFactor(scope, cards, params)
