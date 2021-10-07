from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class F2(Regularizer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


class NullRegularizer(Regularizer):
    def __init__(self, weight: float):
        super().__init__()

    def forward(self, factors):
        return 0
