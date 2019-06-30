import time

import numpy as np
import pandas as pd

from apogee.models import GraphicalModel
from apogee.models.variables import DiscreteVariable

from apogee.io.read import read_json, read_hugin


with open("examples/data/alarm.net", "r") as file:
    model = read_hugin(file.read())

print(model.predict())
