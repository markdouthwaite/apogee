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

from typing import Optional, Any

from tornado.web import Application
from tornado.ioloop import IOLoop

from .handlers import QueryHandler, HealthHandler


class ApogeeServer(Application):
    def __init__(
        self,
        model: "GraphicalModel",
        *args: Optional[Any],
        port: int = 8080,
        address: str = "127.0.0.1",
        ioloop: IOLoop = None,
        subpath: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """A wrapper for exposing an Apogee Server."""

        self._port = port
        self._address = address
        self._ioloop = ioloop
        self._subpath = subpath

        handlers = [
            (f"/{subpath}/health" if subpath else r"/health", HealthHandler),
            (
                f"/{subpath}/query" if subpath else r"/query",
                QueryHandler,
                dict(model=model),
            ),
        ]

        super().__init__(handlers, *args, **kwargs)

    def run(self) -> None:
        """Start the server's eventloop."""

        self.listen(address=self._address, port=self._port)
        if self._ioloop is None:
            IOLoop.current().start()
        else:
            self._ioloop.start()
