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
