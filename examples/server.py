import logging
from apogee.server import ApogeeServer
from apogee.models import BayesianNetwork

logging.basicConfig(level=logging.DEBUG)

alarm = BayesianNetwork.from_hugin("data/alarm.net")
server = ApogeeServer(alarm)
server.run()
