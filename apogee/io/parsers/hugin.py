import re
import itertools
from collections import OrderedDict

import numpy as np


def read_hugin(filename):
    """Read a HUGIN-formatted file."""

    p = HuginReader().read(filename)
    return p


def deformat(s):
    s = str(s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\t", " ", s)
    return s


class HuginReader:
    """Read and build a BayesianNetwork from a HUGIN file."""

    def __init__(self):
        self._data = OrderedDict()

    def write(self, model: "BayesianNetwork", *args, **kwargs):
        """Write a model as a HUGIN-formatted file."""

        raise NotImplementedError("Writing HUGIN models is not yet supported.")

    def read(self, filename: str) -> dict:
        """Read a HUGIN-formatted file as an Apogee BayesianNetwork."""

        with open(filename, "r") as file:
            data = deformat(file.read().strip())

        nodes, potentials = self._extract(data)

        self.parse_nodes(nodes)
        self.parse_potentials(potentials)

        return self.to_dict()

    def parse_nodes(self, nodes):
        for node in nodes:
            name = node[0].strip()

            self._data[name] = {}

            node_data = [x.strip().split(";") for x in node[1:]][0]
            node_data = [x for x in node_data if len(x) > 0]

            for element in node_data:
                element = element.strip()
                if re.search(r"(.*) = \((.*)\)", element) is not None:
                    element = re.search(r"(.*) = \((.*)\)", element)
                    if element.group(1) == "states":
                        key = element.group(1)
                        value = element.group(2).replace('"', "").strip().split(" ")
                    if element.group(1) == "position":
                        key = element.group(1)
                        value = [
                            int(x)
                            for x in element.group(2)
                            .replace('"', "")
                            .strip()
                            .split(" ")
                        ]

                elif re.search(r"(.*) = \"(.*)\"", element) is not None:
                    element = re.search(r"(.*) = \"(.*)\"", element)
                    key, value = element.group(1), element.group(2)
                else:
                    raise ValueError(
                        "Encountered unknown error in parsing element. '{0}'".format(
                            element
                        )
                    )

                self._data[name][key] = value

    def parse_potentials(self, potentials):
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
            self._data[key]["parents"] = parents
            if "position" not in self._data[key].keys():
                self._data[key]["position"] = [0, 0, 0]

    def to_dict(self) -> dict:
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
                cpt=value["cpt"],
                states=value["states"],
                x=value["position"][0],
                y=value["position"][1],
                z=0,
            )

            for parent in value["parents"]:  # this is important
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
