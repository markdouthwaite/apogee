"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

from typing import Optional, Any

from tornado.web import Application
from tornado.ioloop import IOLoop

from .handlers import QueryHandler, HealthHandler
from .handlers.variables import VariablesListHandler, VariableMetaHandler


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
            (
                f"/{subpath}/vars/list" if subpath else r"/vars/list",
                VariablesListHandler,
                dict(model=model),
            ),
            (
                f"/{subpath}/vars/meta" if subpath else r"/vars/meta",
                VariableMetaHandler,
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
