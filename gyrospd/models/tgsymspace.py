import torch
from geoopt.linalg import batch_linalg as lalg
from gyrospd.config import INIT_EPS, DEVICE
from gyrospd.models.base import KGModel
from gyrospd.utils import productory, trace
from gyrospd.manifolds import SPDManifold
from gyrospd.manifolds.metrics import MetricType

__all__ = ["TgSPDModel", "TgSPDRotationModel", "TgSPDReflectionModel", "TgSPDAttnModel"]


class TgSPDModel(KGModel):
    """Knowledge Graph embedding model that operates on the tangent space of the SPD Manifold"""

    def __init__(self, args):
        super().__init__(args)
        self.manifold = SPDManifold(dims=args.dims, metric=MetricType.from_str(args.metric))

        init_fn = lambda n_points: torch.randn((n_points, args.dims, args.dims)) * INIT_EPS
        self.entities = torch.nn.Parameter(init_fn(args.num_entities), requires_grad=True)
        self.relations = torch.nn.Parameter(init_fn(args.num_relations), requires_grad=True)
        # rel_transforms are n x n symmetric matrices
        self.rel_transforms = torch.nn.Parameter(
            torch.rand((args.num_relations, args.dims, args.dims)) * 2 - 1.0,  # U[-1, 1]
            requires_grad=True
        )

        self.addition = self.addition_hrh if args.use_hrh == 1 else self.addition_rhr
        if args.inverse_tail == 1:
            # performs exponential map and inverse in only one diagonalization
            self.map_tail = lambda tg_tails: lalg.sym_funcm(tg_tails, lambda tensor: torch.reciprocal(torch.exp(tensor)))
        else:
            self.map_tail = self.manifold.expmap_id

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """
        tg_heads = lalg.sym(self.entities[triples[:, 0]])                   # b x n x n
        rel_transforms = lalg.sym(self.rel_transforms[triples[:, 1]])       # b x n x n

        tg_heads = rel_transforms * tg_heads                                # b x n x n
        tg_relations = lalg.sym(self.relations[triples[:, 1]])              # b x n x n

        return self.addition(tg_heads, tg_relations)

    def addition_hrh(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :param relations: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        # applies exp_map and sqrt in only one diagonalization
        entities = lalg.sym_funcm(entities, lambda t: torch.sqrt(torch.exp(t)))
        relations = SPDManifold.expmap_id(relations)

        return SPDManifold.addition_id_from_sqrt(entities, relations)

    def addition_rhr(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :param relations: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        # inverts the order of the addition
        return self.addition_hrh(relations, entities)

    def get_rhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n
        """
        tg_tails = lalg.sym(self.entities[triples[:, 2]])                  # b x n x n
        return self.map_tail(tg_tails)

    def similarity_score(self, lhs, rhs):
        dist, _ = self.manifold.dist(lhs, rhs)
        return -1 * dist ** 2, dist

    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        heads = self.entities[triples[:, 0]]
        rel = self.relations[triples[:, 1]]
        rel_transf = self.rel_transforms[triples[:, 1]]
        tails = self.entities[triples[:, 2]]
        return heads, rel, rel_transf, tails

    def compute_norms(self, points):
        entities = self.manifold.expmap_id(lalg.sym(points.detach()))
        return entities.flatten(start_dim=1).norm(dim=-1)

    def entity_norms(self):
        return self.compute_norms(self.entities)

    def relation_norms(self):
        return self.compute_norms(self.relations)

    def relation_transform_norms(self):
        return self.rel_transforms.detach().flatten(start_dim=1).norm(dim=-1)


class TgSPDIsometryModel(TgSPDModel):
    def __init__(self, args):
        super().__init__(args)
        self.dims = args.dims

        self.spd_init = lambda n_points: torch.randn((n_points, args.dims, args.dims)) * INIT_EPS
        self.entities = torch.nn.Parameter(self.spd_init(args.num_entities), requires_grad=True)
        self.relations = torch.nn.Parameter(self.spd_init(args.num_relations), requires_grad=True)
        self.rel_transforms.requires_grad_(False)

        self.n_isom = args.dims * (args.dims - 1) // 2
        self.isom_init = lambda n: torch.rand((n, self.n_isom)) * 0.5 - 0.25  # U[-0.25, 0.25] radians ~ U[-15°, 15°]
        # the purpose of this function will be to initialize the parameters as:
        #   self.rot_params = torch.nn.Parameter(self.isom_init(args.num_relations), requires_grad=True)

        self.embed_index = self.get_isometry_embed_index(self.dims)

    def get_isometry_embed_index(self, dims):
        """
        Build the index to embed the respective isometries into an n x n identity.

        We store a flattened version of the index. This is, for (i, j) the position of an entry in an
        n x n matrix, we reshape the matrix to a single row of len == n*n, and store the equivalent of
        (i, j) in the flattened version of the matrix

        For n dims we build m isometries, where m = n * (n - 1) / 2

        :param dims: int with number of dimensions
        :return: 1 x m x 4: the initial 1 dimension is just to make the repetion of this index faster
        """
        # indexes := 1 <= i < j < n. Using 1-based notation to make it equivalent to matrix math notation
        indexes = [(i, j) for i in range(1, dims + 1) for j in range(i + 1, dims + 1)]

        embed_index = []
        for i, j in indexes:
            row = []
            for c_i, c_j in [(i, i), (i, j), (j, i), (j, j)]:  # 4 combinations that we care for each (i, j) pair
                flatten_index = dims * (c_i - 1) + c_j - 1
                row.append(flatten_index)
            embed_index.append(row)
        return torch.LongTensor(embed_index).unsqueeze(0).to(DEVICE)  # 1 x m x 4

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """
        isometry_params = self.get_isometry_params()
        all_relation_isometries = self.build_relation_isometry_matrices(isometry_params)  # r x n x n
        rel_isometries = all_relation_isometries[triples[:, 1]]  # b x n x n
        tg_heads = lalg.sym(self.entities[triples[:, 0]])  # b x n x n

        tg_heads = rel_isometries @ tg_heads @ rel_isometries.transpose(-1, -2)
        tg_relations = lalg.sym(self.relations[triples[:, 1]])  # b x n x n
        return self.addition(tg_heads, tg_relations)

    def get_isometry_params(self):
        """
        This method must be implemented by concrete clases where the isometry parameters
        for each relation are computed  and returned as a tensor of r x m x 4 where
            r: num of relations
            m: num of isometries
        :return: tensor of r x m x 4
        """
        raise NotImplementedError()

    def build_relation_isometry_matrices(self, isom_params):
        """
        Builds the rotation isometries as matrices for all available relations
        :param isom_params: r x m x 4
        :return: r x n x n
        """
        # isom_params = self.compute_rotation_params(self.rot_params)  # r x m x 4
        embeded_rotations = self.embed_params(isom_params, self.dims)  # r x m x n x n
        isom_rot = productory(embeded_rotations)  # r x n x n
        return isom_rot

    def embed_params(self, iso_params: torch.Tensor, dims: int) -> torch.Tensor:
        """
        Embeds the isometry params.
        For each isometric operation there are m isometries with 4 params each.
        This method embeds the 4 params into a dims x dims identity, in positions given by self.embed_index

        :param iso_params: b x m x 4, where m = dims * (dims - 1) / 2, which is the amount of isometries
        :param dims: (also called n) dimension of output identities, with params embedded
        :return: b x m x n x n
        """
        bs, m, _ = iso_params.size()
        target = torch.eye(dims, requires_grad=True, device=iso_params.device)
        target = target.reshape(1, 1, dims * dims).repeat(bs, m, 1)  # b x m x n * n
        scatter_index = self.embed_index.repeat(bs, 1, 1)  # b x m x 4
        embed_isometries = target.scatter(dim=-1, index=scatter_index, src=iso_params)  # b x m x n * n
        embed_isometries = embed_isometries.reshape(bs, m, dims, dims)  # b x m x n x n
        return embed_isometries

    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        heads = self.entities[triples[:, 0]]
        rel = self.relations[triples[:, 1]]
        tails = self.entities[triples[:, 2]]
        return heads, rel, tails


class TgSPDRotationModel(TgSPDIsometryModel):
    def __init__(self, args):
        super().__init__(args)
        self.rot_params = torch.nn.Parameter(self.isom_init(args.num_relations), requires_grad=True)

    def get_isometry_params(self):
        """
        :return: tensor of r x m x 4
        """
        return self.compute_rotation_params(self.rot_params)

    def compute_rotation_params(self, params):
        """
        Computes rotation parameters:
        For each entry in params computes:
            R^+ = (cos(x), -sin(x), sin(x), cos(x))
        :param params: r x m
        :return: r x m x 4
        """
        cos_x = torch.cos(params)
        sin_x = torch.sin(params)
        res = torch.stack([cos_x, -sin_x, sin_x, cos_x], dim=-1)
        return res


class TgSPDReflectionModel(TgSPDIsometryModel):
    def __init__(self, args):
        super().__init__(args)
        self.ref_params = torch.nn.Parameter(self.isom_init(args.num_relations), requires_grad=True)

    def get_isometry_params(self):
        """
        :return: tensor of r x m x 4
        """
        return self.compute_reflection_params(self.ref_params)

    def compute_reflection_params(self, params):
        """
        Computes reflection parameters:
        For each entry in params computes:
            R^+ = (cos(x), sin(x), sin(x), -cos(x))
        :param params: b x m
        :return: b x m x 4
        """
        cos_x = torch.cos(params)
        sin_x = torch.sin(params)
        res = torch.stack([cos_x, sin_x, sin_x, -cos_x], dim=-1)
        return res


class TgSPDAttnModel(TgSPDRotationModel, TgSPDReflectionModel):
    def __init__(self, args):
        super().__init__(args)
        self.rel_attn = torch.nn.Parameter(self.spd_init(args.num_relations), requires_grad=True)
        self.attn_scale = 1 / torch.Tensor([self.dims]).to(DEVICE).sqrt()
        self.softmax = torch.nn.Softmax(dim=-1)

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """
        all_isom_rot, all_isom_ref = self.get_isometries()          # r x n x n
        isom_rot = all_isom_rot[triples[:, 1]]                      # b x n x n
        isom_ref = all_isom_ref[triples[:, 1]]                      # b x n x n
        tg_heads = lalg.sym(self.entities[triples[:, 0]])           # b x n x n

        tg_head_rot = isom_rot @ tg_heads @ isom_rot.transpose(-1, -2)
        tg_head_ref = isom_ref @ tg_heads @ isom_ref.transpose(-1, -2)

        tg_attn = lalg.sym(self.rel_attn[triples[:, 1]])            # b x n x n
        tg_attn_head = self.compute_attention(tg_head_rot, tg_head_ref, tg_attn)

        tg_relations = lalg.sym(self.relations[triples[:, 1]])      # b x n x n
        return self.addition(tg_attn_head, tg_relations)

    def compute_attention(self, tg_head_rot, tg_head_ref, tg_attn):
        """
        :param tg_head_rot: b x n x n
        :param tg_head_ref: b x n x n
        :param tg_attn: b x n x n
        :return: b x n x n
        """
        weight_rot = trace(tg_attn @ tg_head_rot, keepdim=True) * self.attn_scale       # b x 1
        weight_ref = trace(tg_attn @ tg_head_ref, keepdim=True) * self.attn_scale       # b x 1
        weights = self.softmax(torch.cat([weight_rot, weight_ref], dim=-1))             # b x 2
        weight_rot = weights[:, 0].view(-1, 1, 1)                                       # b x 1 x 1
        weight_ref = weights[:, 1].view(-1, 1, 1)                                       # b x 1 x 1
        attn_result = weight_rot * tg_head_rot + weight_ref * tg_head_ref               # b x n x n
        # saves the weights for stats
        self.attn_weights = weights.detach()

        return attn_result

    def get_isometries(self):
        """
        Builds the isometries as matrices for all available relations
        :return:
        """
        rot_params = TgSPDRotationModel.get_isometry_params(self)           # r x m x 4
        ref_params = TgSPDReflectionModel.get_isometry_params(self)         # r x m x 4
        return self.build_relation_isometry_matrices(rot_params), self.build_relation_isometry_matrices(ref_params)
