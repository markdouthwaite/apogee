"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

from tornado.web import RequestHandler


class HealthHandler(RequestHandler):
    """A healthcheck handler. Ping it to check your service is alive."""

    def get(self, *args, **kwargs):
        self.write("OK")
