import numpy as np
from .factor import DiscreteFactor


class ClassifierFactor(DiscreteFactor):
    def __init__(self, scope, cards, estimator, **kwargs):
        self.estimator = estimator(**kwargs)
        super().__init__(scope, cards)

    def fit(self, x, y=None):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def reduce(self, *evidence):
        self._parameters = self.estimator.predict_proba(evidence[0].reshape(1, -1))[0]

    def refresh(self):
        self._parameters = np.ones_like(self.parameters)
