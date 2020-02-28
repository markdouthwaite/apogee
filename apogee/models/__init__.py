import os
import pkg_resources

from .base import GraphicalModel
from apogee.models import variables, bayes

SPEC_BASE = pkg_resources.resource_filename("apogee", "models/specs")


_models = {"alarm": {"type": "bayes", "path": os.path.join(SPEC_BASE, "alarm.json")}}


def load_model(name):
    if name in _models:
        type = _models[name]["type"]
        path = _models[name]["path"]

        if type == "bayes":
            with open(path) as model_file:
                return bayes.BayesianNetwork.from_json(model_file.read())
        else:
            raise ValueError("Unknown model type '{type}'.".format(type=type))

    else:
        raise ValueError("Unknown model name '{name}'.".format(name=name))
