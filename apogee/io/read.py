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

import json

from typing import Optional, Any

from apogee.io.parsers import HuginParser
from apogee.models import GraphicalModel
from apogee.models.variables import DiscreteVariable


_FLAVOURS: dict = {"discrete": DiscreteVariable}


def read_hugin(data: str) -> GraphicalModel:
    """Read a HUGIN-formatted string and return corresponding PGM (BN)."""

    return read_dict(HuginParser().parse(data))


def read_json(data: str, **kwargs: Optional[Any]) -> GraphicalModel:
    """Read a JSON-formatted string and return corresponding PGM."""

    return read_dict(json.loads(data, **kwargs))


def read_dict(data: dict, directed: bool = True) -> GraphicalModel:
    """Read a dictionary-structured PGM and return full model object."""

    model = GraphicalModel()
    for key, value in data.items():
        if "type" in value:
            flavour = _FLAVOURS[value["type"]]
            del value["type"]
        else:
            flavour = DiscreteVariable

        if directed and "parents" in value:
            value["neighbours"] = value["parents"]
            del value["parents"]

        model.add(flavour(key, **value))

    return model
