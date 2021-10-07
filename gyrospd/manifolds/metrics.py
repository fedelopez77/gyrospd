from abc import ABC, abstractmethod
from enum import Enum
import torch


class MetricType(Enum):
    """Allowed types of metrics that Siegel manifolds support"""
    RIEMANNIAN = "riem"
    FINSLER_ONE = "fone"
    FINSLER_INFINITY = "finf"

    @staticmethod
    def from_str(label):
        types = {t.value: t for t in list(MetricType)}
        return types[label]


class Metric(ABC):

    def __init__(self, dims: int):
        self.dims = dims

    @abstractmethod
    def compute_metric(self, v: torch.Tensor, keepdim=True) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def get(cls, type_str: str, dims: int):
        metrics_map = {
            MetricType.RIEMANNIAN.value: SpdRiemannianMetric,
            MetricType.FINSLER_ONE.value: SpdFinslerOneMetric,
            MetricType.FINSLER_INFINITY.value: SpdFinslerInfinityMetric,
        }

        return metrics_map[type_str](dims)


class SpdRiemannianMetric(Metric):

    def compute_metric(self, v: torch.Tensor, keepdim=True) -> torch.Tensor:
        """
        Given v_i = log(d_i), with d_i the eigenvalues of the crossratio matrix,
        the Riemannian distance is given by the Riemannian norm of this values
        :param v: b x n: v_i = log(d_i), with d_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Riemannian distance
        """
        res = torch.norm(v, dim=-1, keepdim=keepdim)
        return res


class SpdFinslerOneMetric(Metric):

    def compute_metric(self, v: torch.Tensor, keepdim=True) -> torch.Tensor:
        """
        Given v_i = log(d_i), with d_i the eigenvalues of the crossratio matrix,
        the Finsler distance one (F_{1})is given by the summation of the n/2 higher values,
        minus the summation of the n/2 lower values.

        Since v_i is in descending order: v1 < v2 < ... < vn
        the Distance Fone = [sum_{i=n/2}^n v_i] - [sum_{i=0}^{n/2} v_i]

        :param v: b x n: v_i = log(d_i), with d_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Finsler distance
        """
        lower_half, higher_half = v.chunk(chunks=2, dim=-1)     # if last dim of v is not even, then lower_half
                                                                # will have one more column
        res = higher_half.sum(dim=-1, keepdim=keepdim) - lower_half.sum(dim=-1, keepdim=keepdim)
        return res


class SpdFinslerInfinityMetric(Metric):

    def compute_metric(self, v: torch.Tensor, keepdim=True) -> torch.Tensor:
        """
        Given v_i = log(d_i), with d_i the eigenvalues of the crossratio matrix,
        the Finsler infinity metric is max(v) - min(v)

        Since v_i is in descending order: v1 < v2 < ... < vn
        the Distance Finf = vn - v1

        :param v: b x n: v_i = log(d_i), with d_i the eigenvalues of the crossratio matrix,
                where d_n the largest eigenvalue and d_1 the smallest
        :return: b x 1: Finsler distance
        """
        res = v[:, -1] - v[:, 0]
        if keepdim:
            return res.reshape((-1, 1))
        return res
