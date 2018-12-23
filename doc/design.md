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

## NetworkX

```

net.digraph

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

## NetworkX

```

```