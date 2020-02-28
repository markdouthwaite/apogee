from apogee.server import ApogeeServer
from apogee.models import load_model

alarm = load_model("alarm")
server = ApogeeServer(alarm)
server.run()
