"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

import re
import itertools
from collections import OrderedDict
from typing import Text, Tuple, Dict, Any, List, TextIO

import numpy as np

from apogee.utils import strings


def loads(data: Text) -> Dict[Text, Any]:
    """Load a model as an Apogee-formatted dict from a HUGIN-formatted string"""

    return HuginReader().parse(data)


def load(file: TextIO) -> Dict[Text, Any]:
    """Load a model as an Apogee-formatted dict from a HUGIN-formatted file."""

    with file:
        return loads(file.read())


class HuginReaderError(Exception):
    """Generic HuginReader error."""


class HuginReader:
    """
    A HUGIN parser.

    Provides *limited* parsing capabilities for HUGIN-formatted Bayesian Network files.
    """

    def __init__(self):
        self._data = OrderedDict()
        self._structure_pattern = re.compile(r"(.*) = \((.*)\)")
        self._params_pattern = re.compile(r"(.*) = \"(.*)\"")

    def read(self, filename: Text) -> Dict[Text, Any]:
        """Read a HUGIN-formatted and return in dictionary format."""

        with open(filename, "r") as file:
            return self.parse(file.read())

    def parse(self, data: Text) -> Dict[Text, Any]:
        """Read a HUGIN-formatted string and return in dictionary format."""

        nodes, potentials = self._extract(strings.deformat(data.strip()))

        self._parse_nodes(nodes)
        self._parse_potentials(potentials)

        return self.to_dict()

    def _parse_node_structure(self, element: Text) -> Tuple[Text, List[Text]]:
        matches: re.Match = self._structure_pattern.search(element)

        if matches.group(1) == "states":
            key = matches.group(1)
            value = matches.group(2).replace('"', "").strip().split(" ")

        elif matches.group(1) == "position":
            key = matches.group(1)
            value = [
                int(x)
                for x in matches.group(2)
                    .replace('"', "")
                    .strip()
                    .split(" ")
            ]

        else:
            raise HuginReaderError(f"Failed to process element '{element}'.")

        return key, value

    def _parse_node_params(self, element: Text) -> Tuple[Text, Text]:
        matches: re.Match = self._params_pattern.search(element)
        return matches.group(1), matches.group(2)

    def _parse_node(self, element: Text) -> Tuple[Text, Text]:
        element = element.strip()

        if self._structure_pattern.search(element) is not None:
            key, value = self._parse_node_structure(element)

        elif self._params_pattern.search(element) is not None:
            key, value = self._parse_node_params(element)

        else:
            raise ValueError(
                "Encountered unknown error in parsing element. '{0}'".format(
                    element
                )
            )
        return key, value

    def _parse_nodes(self, nodes: List[List[Text]]) -> None:
        for node in nodes:
            name = node[0].strip()

            self._data[name] = {}

            node_data = [x.strip().split(";") for x in node[1:]][0]
            node_data = [x for x in node_data if len(x) > 0]

            for element in node_data:
                key, value = self._parse_node(element)
                self._data[name][key] = value

    def _parse_potentials(self, potentials):
        for potential in potentials:
            scope, data = potential
            if re.search(r"\((.*)\|", scope) is not None:
                key = re.search(r"\((.*)\|", scope).group(1).strip()
                parents = re.search(r"\|(.*)\)", scope).group(1).strip()
                parents = [x for x in parents.split(" ") if x is not ""]
            else:
                key = re.search(r"\((.*)\)", scope).group(1).strip()
                parents = []
            data = [float(x) for x in re.findall(r"\d+\.\d+", data)]
            if len(parents) > 0:
                pstates = [self._data[x]["states"] for x in parents]
                m = len(list(itertools.product(*pstates)))
                n = len(self._data[key]["states"])
                data = np.array(data).reshape((m, n)).flatten("F")

            self._data[key]["parameters"] = data
            self._data[key]["neighbours"] = parents
            if "position" not in self._data[key].keys():
                self._data[key]["position"] = [0, 0, 0]

    def to_dict(self) -> Dict[Text, Any]:
        data = {}
        for key, value in self._data.items():
            del value["position"]

            data[key] = value
        return data

    def to_extended_dict(self) -> dict:
        nodes = OrderedDict()
        edges = []
        for key, value in self._data.items():
            local = dict(
                name=key,
                params=value["params"],
                states=value["states"],
                x=value["position"][0],
                y=value["position"][1],
                z=0,
            )

            for parent in value["neighbours"]:  # this is important
                edge = dict(start=parent, end=key, weight=1.0, d="true")
                edges.append(edge)
            nodes[key] = local

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _extract(data) -> tuple:
        patterns = re.compile(r"node (.*?) \{(.+?)\}")
        nodes = re.findall(patterns, data)
        patterns = re.compile(r"potential (.*?) \{(.+?)\}")
        potentials = re.findall(patterns, data)
        return nodes, potentials
