import json
from apogee.io.parsers import HuginReader
from apogee.models import GraphicalModel
from apogee.models.variables import DiscreteVariable


_FLAVOURS = {"discrete": DiscreteVariable}


def read_hugin(data: str):
    return read_dict(HuginReader().parse(data))


def read_json(data: str, **kwargs):
    return read_dict(json.loads(data, **kwargs))


def read_dict(data: dict) -> GraphicalModel:
    model = GraphicalModel()
    for key, value in data.items():
        if "type" in value:
            flavour = _FLAVOURS[value["type"]]
            del value["type"]
        else:
            flavour = DiscreteVariable
        model.add(flavour(key, **value))

    return model
