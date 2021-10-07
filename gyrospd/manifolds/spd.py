from typing import Tuple
import torch
from geoopt.manifolds.symmetric_positive_definite import SymmetricPositiveDefinite
from geoopt.linalg import batch_linalg as lalg
from gyrospd.manifolds.metrics import Metric, MetricType
from gyrospd.utils import get_logging


class SPDManifold(SymmetricPositiveDefinite):
    def __init__(self, dims=2, ndim=2, metric=MetricType.RIEMANNIAN):
        super().__init__()
        self.dims = dims
        self.ndim = ndim
        self.metric = Metric.get(metric.value, self.dims)

    def dist(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance between 2 points on the SPD manifold

        M = A^(-1/2) B A^(-1/2)
        UDU^1 = M
        D: matrix with eigenvalues of m in descending order: d1, d2,..., dn, where d1 > d2 > ... > dn
        log_eigs = log([d1, d2, ..., dn])
        And then on log_eigs different metrics can be computed.

        :param a, b: b x n x n: elements in SPD
        :return: distance: b: torch.Tensor with distances between a and b in SPD
        :return: vvd: b x n: torch.Tensor with vector-valued distance between a and b
        """
        inv_sqrt_a = lalg.sym_inv_sqrtm1(a)
        m = inv_sqrt_a @ b @ inv_sqrt_a
        m = lalg.sym(m)         # impose symmetry due to numeric instabilities

        try:
            eigvals, _ = torch.symeig(m, eigenvectors=True)     # eigvalues are in descending order
        except RuntimeError as e:
            log = get_logging()
            log.info(f"ERROR: torch.symeig in SPD dist did not converge. m = {m}")
            raise e

        log_eigvals = torch.log(eigvals)
        result = self.metric.compute_metric(log_eigvals, keepdim=keepdim)
        return result, log_eigvals

    @staticmethod
    def expmap_id(x: torch.Tensor) -> torch.Tensor:
        """
        Performs an exponential map using the Identity as basepoint :math:`\operatorname{Exp}_{Id}(u)`.
        :param: x: b x n x n torch.Tensor point on the SPD manifold
        """
        return lalg.sym_funcm(x, torch.exp)

    @staticmethod
    def logmap_id(y: torch.Tensor) -> torch.Tensor:
        """
        Perform an logarithmic map using the Identity as basepoint :math:`\operatorname{Log}_{Id}(y)`.
        :param: y: b x n x n torch.Tensor point on the tangent space of the SPD manifold
        """
        return lalg.sym_funcm(y, torch.log)

    @staticmethod
    def addition_id(a: torch.Tensor, b: torch.Tensor):
        """
        Performs addition using the Identity as basepoint.

        The addition on SPD using the identity as basepoint is :math:`A \oplus_{Id} B = sqrt(A) B sqrt(A)`.

        :param a: b x n x n torch.Tensor points in the SPD manifold
        :param b: b x n x n torch.Tensor points in the SPD manifold.
        :return: b x n x n torch.Tensor points in the SPD manifold
        """
        sqrt_a = lalg.sym_sqrtm(a)
        return sqrt_a @ b @ sqrt_a

    @staticmethod
    def addition_id_from_sqrt(sqrt_a: torch.Tensor, b: torch.Tensor):
        """
        Performs addition using the Identity as basepoint.
        Assumes that sqrt_a = sqrt(A) so it does not apply the sqrt again

        The addition on SPD using the identity as basepoint is :math:`A \oplus_{Id} B = sqrt(A) B sqrt(A)`.

        :param sqrt_a: b x n x n torch.Tensor points in the SPD manifold
        :param b: b x n x n torch.Tensor points in the SPD manifold.
        :return: b x n x n torch.Tensor points in the SPD manifold
        """
        return sqrt_a @ b @ sqrt_a

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        """
        Random sampling on the manifold.

        The exact implementation depends on manifold and usually does not follow all
        assumptions about uniform measure, etc.
        """
        from_ = kwargs.get("from_", -0.001)
        to = kwargs.get("to", 0.001)
        init_eps = (to - from_) / 2
        dims = self.dims
        perturbation = torch.randn((size[0], dims, dims), dtype=dtype, device=device) * init_eps
        perturbation = lalg.sym(perturbation)
        identity = torch.eye(dims).unsqueeze(0).repeat(size[0], 1, 1)
        return identity + perturbation

    def extra_repr(self) -> str:
        return f"metric={type(self.metric).__name__}"
