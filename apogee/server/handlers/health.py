from tornado.web import RequestHandler


class HealthHandler(RequestHandler):
    def get(self, *args, **kwargs):
        self.write("OK")
