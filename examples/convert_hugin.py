import json
from apogee.io import parsers

hugin = parsers.HuginReader()

data = hugin.read("data/asia.net")

json.dump(data, open("data/asia.json"), indent=4)
