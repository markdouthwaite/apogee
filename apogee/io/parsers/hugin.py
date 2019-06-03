import re
from collections import OrderedDict
import itertools
import numpy as np


def load_hugin(filename):
    p = HuginParser().parse(filename)
    return p


def deformat(s):
    s = str(s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\t", " ", s)
    return s


class HuginParser(object):
    def __init__(self):
        self._data = OrderedDict()

    def write(self, graph, filename):
        f = open(filename, "w")
        # for node in graph:
        #     pass
        raise NotImplementedError("Writing HUGIN models is not yet supported.")

    def parse(self, filename, filepath=""):
        f = open(filepath + filename, "r")
        s = deformat(f.read().strip())

        p = re.compile(r"node (.*?) \{(.+?)\}")
        nodes = re.findall(p, s)
        p = re.compile(r"potential (.*?) \{(.+?)\}")
        potentials = re.findall(p, s)
        self.parse_nodes(nodes)
        self.parse_potentials(potentials)
        return self._to_apg()

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

    def _to_apg(self):
        data = {}
        for key, value in self._data.items():
            del value["position"]
            data[key] = value
        return data

    def _to_d3(self):
        pass

    def _to_adict(self):
        nodes = OrderedDict()
        edges = []
        for key, value in self._data.items():
            local = {}
            local["name"] = key
            local["cpt"] = value["cpt"]
            local["states"] = value["states"]
            # local['parents'] = value['parents']
            local["x"] = value["position"][0]
            local["y"] = value["position"][1]
            local["z"] = 0
            for parent in value["parents"]:  # this is important
                edge = {}
                edge["start"] = parent
                edge["end"] = key
                edge["weight"] = 1.0
                edge["d"] = "true"
                edges.append(edge)
            nodes[key] = local

        return {"nodes": nodes, "edges": edges}
