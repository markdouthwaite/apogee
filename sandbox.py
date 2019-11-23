import time

import numpy as np
import pandas as pd

from apogee.models import GraphicalModel
from apogee.models.variables import DiscreteVariable


class CPTConfig:
    def __init__(self, states: list, dependencies: list, parameters: list):
        self.states = states
        self.dependencies = dependencies
        self.parameters = parameters


class NodeConfig:

    dist = {
        "cpt": CPTConfig
    }

    def __init__(self, name: str, type: str = "cpt", distribution: dict = None, **config):
        self.name = name
        self.type = type
        self.distribution = self.dist[type](**distribution)


class GraphConfig:
    def __init__(self, nodes: list, **metadata: dict):
        self.nodes = nodes
        self.metadata = metadata

    def from_yaml

# from apogee.io.read import read_json, read_hugin
#
#
# with open("examples/data/alarm.net", "r") as file:
#     model = read_hugin(file.read())
#
# print(model.predict())
