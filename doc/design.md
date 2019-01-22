# Basic API
```
from apogee import BayesianNetwork, DiscreteVariable as Variable

with BayesianNetwork("simple-bayesian-network") as net:
    net.add(Variable("cloudy", states=["true", "false"])))
    net.add(Variable("sprinkler", states=["true", "false"]), parents=["cloudy"])
    net.add(Variable("rain"), parents=["cloudy"])
    net.add(Variable("wetgrass"), parents=["sprinkler", "rain"])

net.compile()  # check model structure, build underlying FactorGraph. Silent call in above context manager.

net.fit(x)  # x -> [{"cloudy": "true", ... "wetgrass": "true"}, ... ]
net.predict(x)  # x -> [{...}] ->> {"cloudy": {"true": 0.12341, "false": 0.87659}, ...}

```

# Sci-kit Learn Classifiers

```
from sklearn.tree import DecisionTreeClassifier
from apogee import BayesianNetwork, ClassifierVariable, Variable

with BayesianNetwork() as network:
    network.add(ClassifierVariable("iris-variant", model=DecisionTreeClassifier, states=["i-setosa", "i-versicolor", "i-virginica"]))
    network.add(Variable("light-levels", states=["good", "poor"])
    network.add(Variable("temperature", states=["high", "normal", "low", "below freezing"])
    network.add(Variable("water-levels", states=["high", "normal", "low"], parents=["temperature"])
    network.add(Variable("plant-health", states=["fabulous", "good", "poor"], parents=["water-levels", "light-levels", "iris-variant"]

network.fit()

network.predict({"iris-variant": [5.7, 2.5, 5.0, 2.0], "light-levels": "good")["plant-health"]

{"plant-health": {"fabulous": 0.87, "good": 0.12, "poor": 0.01}}

```

## NetworkX

```

net.graph

```

# I/O
```
from apogee.io import read_json, read_hugin

net = read_json("bayes-net.json")
net = read_hugin("bayes-net.net")

```

# REST service

```
from apogee.rest import RESTService 

service = RESTService(model=net, port=8080, address="127.0.0.1")
service.run() 
```

# Low-level API

```

graph = FactorGraph()
graph.add(DiscreteFactor([0, 1], [2, 2], ...))
graph.add(DiscreteFactor([1, 2], ...))

graph.query(...)

```
