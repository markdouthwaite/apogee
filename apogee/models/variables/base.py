"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseVariable(ABC):
    def __init__(
        self,
        name: str,
        neighbours: Optional[List[str]] = None,
        graph: "GraphicalModel" = None,
    ):
        self.name = name
        self.neighbours = neighbours or []
        self.graph = graph

    def __call__(self, graph: "GraphicalModel"):
        self.graph = graph
        return self

    @property
    def scope(self) -> List[str]:
        return [self.name, *self.neighbours]

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def factor(self):
        pass

    @property
    @abstractmethod
    def flavour(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, self.name)
