"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

import json

from typing import Optional, Any

from tornado.web import RequestHandler


class QueryHandler(RequestHandler):
    """
    A handler for managing requests to a Probabilistic Graphical Model.
    """

    def initialize(self, model: "GraphicalModel", **kwargs: Optional[Any]) -> None:
        """Setup the handler."""

        self._model = model
        super().initialize(**kwargs)

    @staticmethod
    def _format_response(dist: dict) -> str:
        """
        Process the response to strip any nasty numpy data types before serialising.
        """
        dist = {k: {s: float(p) for s, p in v.items()} for k, v in dist.items()}
        return json.dumps(dist)

    def post(self, *args: Optional[Any], **kwargs: Optional[Any]) -> None:
        """
        Execute a query against the provided model.

        {
            "marginals": ["asia"],
            "evidence": {"SMOKER": yes}
        }
        """

        data = self.request.body.decode("utf-8")
        if len(data) > 0:
            payload = json.loads(data)
            evidence = payload.get("evidence", {})
            marginals = payload.get("marginals", None)
            evidence = [(k, v) for k, v in evidence.items()]
            dist = self._model.predict(x=evidence, marginals=marginals)
            # dist = {k: v for k, v in dist.items() if k in marginals}
        else:
            dist = self._model.predict()

        self.write(self._format_response(dist))
