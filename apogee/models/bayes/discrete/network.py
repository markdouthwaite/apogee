import json

import networkx as nx

from apogee.factors import FactorSet
from apogee.inference import JunctionTree
from apogee.io.parsers.hugin import HuginReader

from .discretevariable import DiscreteVariable


def read_hugin(*args, **kwargs) -> "BayesianNetwork":
    """Read a HUGIN file and return a BN."""
    reader = HuginReader()
    data = reader.read(*args, **kwargs)
    return BayesianNetwork.from_dict(data)


class BayesianNetwork:
    """
    A wrapper for building and querying Bayesian Networks!

    References
    ----------
    [1] D. Koller, N. Freidman: Probabilistic Graphical Models, Principles and
        Techniques
    [2] F. Jensen: Bayesian Networks

    """

    _algorithms: dict = {"exact-bp": JunctionTree}
    # re-implement MCMC, variational and 'fast' exact-bp.

    def __init__(self, atype: str = "exact-bp"):
        """
        Initialise the network!

        Parameters
        ----------
        atype: str, optional
            The type of inference algorithm to use. Select from: exact-bp.

        """

        self._atype = atype
        self._variables = {}
        self._algorithm = self._algorithms[atype]

    def id(self, variable: DiscreteVariable) -> int:
        """
        Given a variable, return the index of that variable in the available index list.
        """

        return list(self._variables.keys()).index(variable.name)

    def name(self, variable_id: int) -> str:
        """Given a variable index (id), return the corresponding variable object."""

        return list(self._variables.keys())[variable_id]

    def add(self, variable: DiscreteVariable) -> None:
        """Add a given variable to the network."""

        self._variables[variable.name] = variable

    def fit(self, frame: "DataFrame") -> None:
        for name, variable in self._variables.items():
            variable.fit(frame[variable.scope].values)

    def predict(self, x: dict = None, y: set = None) -> dict:
        """
        Generate predictions (posterior marginal distributions) for each variable in
        the network.

        Parameters
        ----------
        x: dict, optional
            A dictionary containing 'evidence' of the state of the network. This should
            be key: value pairs, where keys correspond to the names of variables, and
            the values are the states or observed value of that variable.
        y: set, optional
            An optional set containing only the names of variables for which you
            wish to retrieve the marginals.
        Returns
        -------
        out: dict
            A dictionary, mapping names to marginal distributions. For a discrete
            variable, this would look like:
            {"var0": {"true": 0.5, "false": 0.5}}

        """

        # Todo: this is far too expensive!
        factors = FactorSet(*[var.factor.copy() for var in self._variables.values()])
        if isinstance(self._algorithm, type):
            self._algorithm = self._algorithm.from_factors(factors)

        if x is not None:
            obs = [
                [self.id(self._variables[k]), self.state_index(self._variables[k], v)]
                for k, v in x.items()
            ]
            self._algorithm.update_observations(obs)

        # this needs to be hidden...
        self._algorithm.propagate()
        self._algorithm.calibrate()

        marginals = {}

        if y is not None:
            v = [v for v in factors.vars if self.name(v) in y]
        else:
            v = factors.vars

        for marginal in self._algorithm.marginals(*v):
            name = self.name(marginal.scope[0])
            variable = self._variables[name]
            current_marginals = {}
            for i, p in enumerate(marginal.normalise().parameters):
                current_marginals.update(**{variable.states[i]: p})
            marginals.update(**{name: current_marginals})

        self._algorithm = self._algorithm.from_factors(factors)

        return marginals

    def compile(self) -> None:
        """
        Compile the model. This will generate corresponding factors for each variable.
        """

        for variable in self._variables.values():
            variable.build_factor(self)

    @staticmethod
    def state_index(variable: DiscreteVariable, state: str) -> int:
        """Get the index of a given state for a variable."""

        # TODO: this is not generic - will not work for CLGs etc.
        return variable.states.tolist().index(state)

    def to_digraph(self) -> nx.DiGraph:
        """Translate the network structure into a NetworkX DiGraph structure."""

        graph = nx.DiGraph()
        for variable in self._variables.values():
            graph.add_node(variable.name, name=variable.name)

        for _, variable in self._variables.items():
            for parent in variable.parents:
                graph.add_edge(parent, variable.name)

        return graph

    def to_dict(self) -> dict:
        """Translate the network structure into a dictionary format."""

        data = {"algorithm": self._atype}
        for name, variable in self._variables.items():
            data[name] = dict(
                states=variable.states.tolist(),
                parents=variable.parents.tolist(),
                parameters=variable.parameters,
            )
        return data

    def to_json(self, **kwargs) -> str:
        """Translate the network structure into a JSON-structured format."""

        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianNetwork":
        """Initialise a BayesianNetwork object from a dictionary."""

        with cls(data.get("algorithm", "exact-bp")) as network:
            for variable, config in data.items():
                network.add(DiscreteVariable(name=variable, **config))

        return network

    @classmethod
    def from_json(cls, data: str, **kwargs: any):
        """Initialise a BayesianNetwork object from a JSON-structured string."""

        data = json.loads(data, **kwargs)
        return cls.from_dict(data)

    @classmethod
    def from_hugin(cls, data: str):
        """Initialise a BayesianNetwork object from a HUGIN-structured string."""

        return cls.from_dict(HuginReader().parse(data))

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

    def __repr__(self):
        return "{0}(n={1})".format(type(self).__name__, len(self))
