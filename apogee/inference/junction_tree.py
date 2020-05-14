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

from typing import Tuple, List, Generator

import networkx as nx
import numpy as np

from apogee.core import get_elimination_ordering, union1d, difference1d
from apogee.utils.typing import FactorLike, FactorSetLike


class JunctionTree:
    """
    An implementation of the Junction Tree algorithm.

    The algorithm performs belief propagation on graphs for the purposes of
    computing exact marginal probabilities for variables in a PGM.

    This implementation is based on the implementation found in [1].

    For those without access to the book, the basic process can be found in [2].

    References
    ----------
    [1]
    [2] https://en.wikipedia.org/wiki/Junction_tree_algorithm

    # Todo: Complete an optimisation pass over this. Lots to tighten up.

    """

    def __init__(self):
        self.graph = nx.Graph()

    def add(
        self, variable: int, factor: FactorLike, tau: List[int], neighbours: List[int]
    ) -> None:
        """Add a variable to the tree."""

        self.graph.add_node(variable, factor=factor, tau=tau)
        for node, attrs in self.graph.nodes.items():
            if node != variable and attrs["tau"] in neighbours:
                messages = {(variable, node): None, (node, variable): None}
                self.graph.add_edge(variable, node, messages=messages)

    def initialise(self, factors: List[FactorLike]) -> "JunctionTree":
        """Initialise the tree given a set of factors."""

        factors = [factor.copy() for factor in factors]
        used = []
        for i, attrs in self.graph.nodes.items():
            factor = attrs["factor"]
            for other in factors:
                if len(np.intersect1d(factor.scope, other.scope)) == len(other.scope):
                    factor *= other
                    used.append(other)
            factors = [x for x in factors if x not in used]
            self.graph.nodes[i]["factor"] = factor
            self.graph.nodes[i]["cached"] = factor.copy()
        return self

    def calibrate(self) -> None:
        """Calibrate the nodes on the tree."""

        for node, attrs in self.graph.nodes.items():
            factor = attrs["factor"]

            for (source, target) in nx.edges(self.graph, node):
                if target != source:
                    factor *= self._message(target, source)

            attrs.update(factor=factor)

    def propagate(self) -> None:
        """Propagate belief across the tree."""

        while self._message_count() < (2.0 * len(self.graph.edges)):
            for (source, target) in self.graph.edges.keys():
                if self._can_send(source, target):
                    self._send_message(source, target)

                elif self._can_send(target, source):
                    self._send_message(target, source)

    def update_observations(self, observations: List[List[int]]) -> None:
        """
        Update the observation state of the tree.

        Parameters
        ----------
        observations: list
            A list of observations of the form [[var: int, obs: int], ..., [...]].
            Where 'obs' is the observed evidence for the state of variable 'var'.

        """

        observations = observations or []
        for variable, state in observations:
            for node in self.graph.nodes:
                factor = self.graph.nodes[node]["factor"]
                if variable in factor.scope:
                    factor = factor.reduce([variable, state])
                    self.graph.nodes[node]["factor"] = factor

    def reset_observations(self) -> None:
        """Reset the observation state of the tree."""

        for node in self.graph.nodes:
            self.graph.nodes[node]["factor"] = self.graph.nodes[node]["cached"].copy()

    def marginal(self, variable: int) -> FactorLike:
        """Compute the marginal distribution for the given variable."""

        for factor in self.factors:
            if variable in factor.scope:
                return factor.marginalise(*np.setdiff1d(factor.scope, [variable]))

        raise ValueError(
            "Variable '{0}' was not found in the provided tree.".format(variable)
        )

    def marginals(self, *variables: Tuple[int]) -> FactorLike:
        """Compute the marginal distribution for a collection of variables."""

        for variable in variables:
            yield self.marginal(variable)

    def _can_send(self, source: int, target: int) -> bool:
        """Determine if messages can be sent from source node to target node."""

        if self._message(source, target) is None:
            neighbours = list(nx.neighbors(self.graph, source))
            n_neighbours = len(neighbours)
            n_received = sum(
                self._has_received(neighbour, source) for neighbour in neighbours
            )

            if n_received == n_neighbours - 1 and not self._has_received(
                target, source
            ):
                return True
            elif n_received == n_neighbours and self._has_received(target, source):
                return True

        return False

    def _message(self, source: int, target: int) -> FactorLike:
        """Get the message sent between the source and target node."""

        return self.graph.edges[(source, target)]["messages"][(source, target)]

    def _has_received(self, source: int, target: int) -> bool:
        """Check if a message was sent between the source and target node."""

        return True if self._message(source, target) is not None else False

    def _send_message(self, source: int, target: int) -> None:
        """Send a message between the source and target node."""

        source_factor = self.graph.nodes[source]["factor"].copy()
        target_factor = self.graph.nodes[target]["factor"].copy()

        source_scope = source_factor.scope
        target_scope = target_factor.scope

        for source, other in nx.edges(self.graph, source):
            if other != target and self._message(other, source) is not None:
                source_factor *= self._message(other, source)

        targets = [
            x
            for x in source_scope
            if x not in [x for x in source_scope if x in target_scope]
        ]
        source_factor = source_factor.marginalise(*targets)

        self.graph.edges[(source, target)]["messages"][(source, target)] = source_factor

    def _message_count(self) -> int:
        """Calculate the total number of messages sent."""

        messages = 0
        for k, v in nx.get_edge_attributes(self.graph, "messages").items():
            for _, z in v.items():
                if z is not None:
                    messages += 1

        return messages

    @property
    def factors(self) -> Generator[FactorLike, None, None]:
        """Yield factors in the tree."""

        for node in self.graph.nodes.values():
            yield node["factor"]

    @classmethod
    def from_factors(cls, factor_set: FactorSetLike) -> "JunctionTree":
        """Create a JT from a provided FactorSet object."""

        tree = cls()

        factor_scopes = [x.scope.tolist() for x in factor_set]
        for variable, _ in zip(*get_elimination_ordering(factor_set.adjacency_matrix)):
            current_factor_scope = union1d(
                *[scope for scope in factor_scopes if variable in scope]
            )
            current_tau_scope = difference1d(current_factor_scope, [variable]).tolist()

            current_neighbour_scopes = [x for x in factor_scopes if variable in x]

            tree.add(
                variable,
                factor_set.new_factor(current_factor_scope),
                current_tau_scope,
                current_neighbour_scopes,
            )

            factor_scopes = [scope for scope in factor_scopes if variable not in scope]

            if current_tau_scope not in factor_scopes:
                factor_scopes.append(current_tau_scope)

        tree.initialise(factor_set.factors)

        return tree
