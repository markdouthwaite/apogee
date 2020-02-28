from tornado.web import Application
from tornado.ioloop import IOLoop

from .handlers import QueryHandler, HealthHandler


class ApogeeServer(Application):
    def __init__(
        self, model, *args, port=8080, address="127.0.0.1", ioloop=None, **kwargs
    ):
        self._port = port
        self._address = address
        self._ioloop = ioloop

        handlers = [
            (r"/health", HealthHandler),
            (r"/query", QueryHandler, dict(model=model)),
        ]

        super().__init__(handlers, *args, **kwargs)

    def run(self) -> None:
        self.listen(address=self._address, port=self._port)
        if self._ioloop is None:
            IOLoop.current().start()
        else:
            self._ioloop.start()
