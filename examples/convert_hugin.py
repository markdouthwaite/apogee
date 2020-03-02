import json
from apogee.io import parsers

hugin = parsers.HuginParser()

data = hugin.read("data/asia.net")

json.dump(data, open("data/asia.json"), indent=4)
