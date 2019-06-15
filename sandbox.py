from apogee.models import BayesianNetwork, Variable


with BayesianNetwork() as net:
    net.add(Variable("cloudy", states=["true", "false"],
                     parameters=[0.4, 0.6]))

    net.add(Variable("sprinkler", states=["true", "false"], parents=["cloudy"],
                     parameters=[[0.1, 0.9],
                                 [0.7, 0.3]]))

    net.add(Variable("rain", states=["true", "false"], parents=["cloudy"],
                     parameters=[[0.8, 0.2],
                                 [0.3, 0.7]]))

    net.add(Variable("wetgrass", states=["true", "false"], parents=["sprinkler", "rain"],
                     parameters=[[0.9, 0.1],
                                 [0.4, 0.6],
                                 [0.7, 0.3],
                                 [0.1, 0.9]]))

print(net.to_yaml())

