from apogee.models.bayes import BayesianNetwork, DiscreteVariable

with BayesianNetwork() as bn:
    bn.add(DiscreteVariable("a", ["a0", "a1"], parameters=[0.1, 0.9]))
    bn.add(
        DiscreteVariable(
            "b", ["b0", "b1"], parents=["a"], parameters=[0.8, 0.2, 0.45, 0.55]
        )
    )
    bn.add(
        DiscreteVariable(
            "c", ["c0", "c1"], parents=["a"], parameters=[0.3, 0.7, 0.45, 0.55]
        )
    )

print(bn.predict({"a": "a0"}))
