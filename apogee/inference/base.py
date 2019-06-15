from abc import ABC, abstractmethod


class InferenceAlgorithm(ABC):
    """Abstract class defining the interface for Inference Algorithms."""

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def marginal(self, arg):
        pass

    @abstractmethod
    def marginals(self, *args):
        pass

    @abstractmethod
    def execute(self):
        pass

    @property
    @abstractmethod
    def factors(self):
        pass

    @classmethod
    @abstractmethod
    def from_factors(cls, *args, **kwargs):
        pass
