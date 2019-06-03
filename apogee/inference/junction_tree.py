import numpy as np
import networkx as nx
from apogee import get_elimination_ordering, union1d, difference1d


class JunctionTree(object):
    def __init__(self):
        self.graph = nx.Graph()

    def add(self, variable, factor, tau, neighbours=None):
        self.graph.add_node(variable, factor=factor, tau=tau)
        if neighbours is not None:
            for node, attrs in self.graph.nodes.items():
                if node != variable:
                    if attrs["tau"] in neighbours:
                        self.graph.add_edge(
                            variable,
                            node,
                            messages={(variable, node): None, (node, variable): None},
                        )

    def initialise(self, factors):
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

    def calibrate(self):
        for node, attrs in self.graph.nodes.items():
            factor = attrs["factor"]

            for (source, target) in nx.edges(self.graph, node):
                if target != source:
                    factor *= self._message(target, source)

            attrs.update(factor=factor)

    def propagate(self):
        while self._message_count() < (2.0 * len(self.graph.edges)):
            for (source, target) in self.graph.edges.keys():
                if self._can_send(source, target):
                    self._send_message(source, target)

                elif self._can_send(target, source):
                    self._send_message(target, source)

    def update_observations(self, observations=None):
        observations = observations or []
        for variable, state in observations:
            for node in self.graph.nodes:
                factor = self.graph.nodes[node]["factor"]
                if variable in factor.scope:
                    factor = factor.reduce([variable, state])
                    self.graph.nodes[node]["factor"] = factor

    def reset_observations(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["factor"] = self.graph.nodes[node]["cached"].copy()

    def marginal(self, variable):
        for factor in self.factors:
            if variable in factor.scope:
                return factor.marginalise(*np.setdiff1d(factor.scope, [variable]))

    def marginals(self, *variables):
        for variable in variables:
            yield self.marginal(variable)

    def _can_send(self, source, target):
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

    def _message(self, source, target):
        return self.graph.edges[(source, target)]["messages"][(source, target)]

    def _has_received(self, source, target):
        return True if self._message(source, target) is not None else False

    def _send_message(self, source, target):
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

    def _message_count(self):
        messages = 0
        for k, v in nx.get_edge_attributes(self.graph, "messages").items():
            for _, z in v.items():
                if z is not None:
                    messages += 1
        return messages

    @property
    def factors(self):
        for node in self.graph.nodes.values():
            yield node["factor"]

    @classmethod
    def from_factors(cls, factor_set):
        tree = cls()

        factor_scopes = [x.scope.tolist() for x in factor_set]
        for variable, reduced_scope in zip(
            *get_elimination_ordering(factor_set.adjacency_matrix)
        ):
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
