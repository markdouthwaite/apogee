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

from .undirected import UndirectedModel
from .directed import BayesianNetwork, DirectedModel

# import os
# import pkg_resources

# from .base import GraphicalModel
# from apogee.models.variables import DiscreteVariable
#
# BayesianNetwork = GraphicalModel
#
#
# SPEC_BASE: str = pkg_resources.resource_filename("apogee", "models/specs")
#
#
# _models: dict = {
#     "alarm": {"type": "bayes", "path": os.path.join(SPEC_BASE, "alarm.json")}
# }


# def load_model(name):
#     if name in _models:
#         type = _models[name]["type"]
#         path = _models[name]["path"]
#
#         if type == "bayes":
#             with open(path) as model_file:
#                 return read_json(model_file.read())
#         else:
#             raise ValueError("Unknown model type '{type}'.".format(type=type))
#
#     else:
#         raise ValueError("Unknown model name '{name}'.".format(name=name))

#
# __all__ = ["BayesianNetwork", "GraphicalModel"]
