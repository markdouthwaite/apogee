import json
import networkx as nx
from .variable import Variable
from ...factors import FactorSet
from ...inference import JunctionTree
from apogee.io.parsers import load_hugin


class BayesianNetwork(object):

    _inference_engines = {"exact-bp": JunctionTree}

    _variable_types = {"discrete": Variable}

    def __init__(self, engine="exact-bp"):
        """

        Parameters
        ----------
        engine

        """

        self._variables = {}
        self._engine = self._inference_engines[engine]

    def id(self, variable):
        return list(self._variables.keys()).index(variable.name)

    def name(self, variable_id):
        return list(self._variables.keys())[variable_id]

    def add(self, variable):
        self._variables[variable.name] = variable

    def predict(self, x: dict = None):
        if isinstance(self._engine, type):
            factors = FactorSet(
                *[var.factor.copy() for var in self._variables.values()]
            )
            self._engine = self._engine.from_factors(factors)
        else:
            factors = FactorSet(*list(self._engine.factors))

        if x is not None:
            obs = [
                [self.id(self._variables[k]), self.state_index(self._variables[k], v)]
                for k, v in x.items()
            ]
            self._engine.update_observations(obs)

        self._engine.propagate()
        self._engine.calibrate()

        marginals = {}
        for marginal in self._engine.marginals(*factors.vars):
            name = self.name(marginal.scope[0])
            variable = self._variables[name]
            current_marginals = {}
            for i, p in enumerate(marginal.normalise().parameters):
                current_marginals.update(**{variable.states[i]: p})
            marginals.update(**{name: current_marginals})
        self._engine.reset_observations()
        return marginals

    def compile(self):
        for variable in self._variables.values():
            variable.build_factor(self)

    def state_index(self, variable, state):
        return variable.states.tolist().index(state)

    def graph(self):
        graph = nx.DiGraph()
        for variable in self._variables.values():
            graph.add_node(variable.name, name=variable.name)

        for key, variable in self._variables.items():
            for parent in variable.parents:
                graph.add_edge(parent, variable.name)

        return graph

    def to_dict(self):
        data = {}
        for name, variable in self._variables.items():
            data[name] = dict(
                states=variable.states,
                parents=variable.parents,
                parameters=variable.parameters,
            )
        return data

    def to_json(self, filename, **kwargs):
        json.dump(self.to_dict(), open(filename, "r"), **kwargs)

    @classmethod
    def from_dict(cls, data, engine="exact-bp"):
        with cls(engine) as network:
            for variable, config in data.items():
                network.add(Variable(name=variable, **config))
        return network

    @classmethod
    def from_json(cls, filename, engine="exact-bp", **kwargs):
        data = json.load(open(filename, "r"), **kwargs)
        return cls.from_dict(data, engine=engine, **kwargs)

    @classmethod
    def from_hugin(cls, filename, engine="exact-bp", **kwargs):
        data = load_hugin(filename)
        return cls.from_dict(data, engine=engine, **kwargs)

    def __len__(self):
        return len(self._variables)

    def __iter__(self):
        for variable in self._variables.values():
            yield variable

    def __getitem__(self, item):
        return self._variables[item]

    def __setitem__(self, key, value):
        self._variables[key] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.compile()
