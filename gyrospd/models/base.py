from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class KGModel(nn.Module, ABC):
    """
    Knowledge Graph embedding model that operates on SPD Manifold, but can easily be extended
    to other manifolds as well
    """
    def __init__(self, args):
        super().__init__()
        self.bias_head = nn.Embedding(args.num_entities, 1)
        self.bias_head.weight.data = torch.zeros((args.num_entities, 1))
        self.bias_tail = nn.Embedding(args.num_entities, 1)
        self.bias_tail.weight.data = torch.zeros((args.num_entities, 1))
        self.bias_head.weight.requires_grad = self.bias_tail.weight.requires_grad = args.train_bias

        self.all_entities_ids = torch.arange(start=0, end=args.num_entities, dtype=torch.long).unsqueeze(dim=-1)
        self.eval_batch_size = args.eval_batch_size

    @abstractmethod
    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x *: points in the space to compute distance
        """
        pass

    @abstractmethod
    def get_rhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x *: points in the space to compute distance
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs, rhs):
        """
        Usually sim = -d(lhs, rhs)^2
        :param lhs: b x *
        :param rhs: b x *
        :return: b x 1
        """
        pass

    def forward(self, triples):
        """
        sigma(h, r, t) = -d(lhs, rhs)^2 + b_h + b_t
        :param triples: b x 3: (head, relation, tail)
        :return score: b x 1
        """
        lhs = self.get_lhs(triples)
        lhs_bias = self.bias_head(triples[:, 0])
        rhs = self.get_rhs(triples)
        rhs_bias = self.bias_tail(triples[:, 2])

        sim_score, dist = self.similarity_score(lhs, rhs)
        return sim_score + lhs_bias + rhs_bias, dist

    def forward_eval(self, triples):
        """
        Evaluates the model according to (h, r, ?). Ignores tail entities and computes scores from (h,r) to
        all available entities.
        :param triples: b x 3: (head, relation, tail)
        :return score: b x 1
        """
        lhs = self.get_lhs(triples).unsqueeze(1)    # b x 1 x * x n x n With the added dim the rest is broadcasted
        lhs_bias = self.bias_head(triples[:, 0])    # b x 1
        rhs = self.get_rhs(self.all_entities_ids.repeat(1, 3))  # num_entities x * x n x n
        rhs_bias = self.bias_tail.weight            # num_entites x 1

        sim_score, _ = self.similarity_score(lhs, rhs)      # b x num_entities x 1
        return sim_score.squeeze(-1) + lhs_bias + rhs_bias.transpose(-1, -2)

    def evaluate(self, triples):
        """
        For evaluation, the model has to score (head, relation, ?) versus all entities,
        and also (head, relation, tail) versus the actual test tail.
        :param triples: b x 3: (head, relation, tail)
        :return: all_scores: b x num_entities: scores of (head, relation, ?) versus all entities
        :return: target_scores: b x 1: scores of (head, relation, tail) versus test entity
        """
        all_result = self.forward_eval(triples)     # b x num_entities
        # uses tail as index to get target score
        target_result = all_result.gather(dim=-1, index=triples[:, 2].unsqueeze(-1))
        return all_result, target_result

    @abstractmethod
    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        pass

    @abstractmethod
    def entity_norms(self):
        """:return: tensor with one element"""
        pass

    @abstractmethod
    def relation_norms(self):
        """:return: tensor with one element"""
        pass

    @abstractmethod
    def relation_transform_norms(self):
        """:return: tensor with one element"""
        pass
