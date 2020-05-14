"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

import json
from typing import Optional, Any

from tornado.web import RequestHandler


class VariablesListHandler(RequestHandler):
    """A handler that returns a list of the names of variables in the model."""

    def initialize(self, model: "GraphicalModel", **kwargs: Optional[Any]) -> None:
        """Setup the handler."""

        self._model = model
        super().initialize(**kwargs)

    def get(self, *args, **kwargs):
        names = [v for v in self._model.variables]
        data = json.dumps({"vars": names})
        self.write(data)
