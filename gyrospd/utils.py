import random
import torch
import numpy as np
import logging
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    log.parent.disabled = True
    log.propagate = False
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def write_results_to_file(results_file: str, results_data: dict):
    results_data["timestamp"] = [datetime.now().strftime("%Y%m%d%H%M%S")]
    file = Path(results_file)
    pd.DataFrame.from_dict(results_data).to_csv(file, mode="a", header=not file.exists())


def get_inverted_triples(triples, num_relations):
    """
    :param triples: b x 3, numpy array with (head, rel, tail)
    :param num_relations: int
    :return: inverted_triples: b x 3, numpy array with (tail, rel + num_relations, head)
    """
    inverse_rels = triples[:, 1] + num_relations
    inverted_triples = np.stack((triples[:, 2], inverse_rels, triples[:, 0]), axis=1)
    return inverted_triples


def compute_metrics(ranking: torch.Tensor):
    """
    :param ranking: tensor of b x 1 with ranking of each evaluation triple. 1 is the highest rank.
    :return: dict with keys: ["MRR", "MR", "HR@1", "HR@3", "HR@10"], and float values
    """
    metrics = {"MR": ranking.mean().item(), "MRR": (1. / ranking).mean().item()}
    for k in (1, 3, 10):
        metrics[f'HR@{k}'] = (ranking <= k).float().mean().item()
    return metrics


def avg_side_metrics(side_metrics):
    """
    :param side_metrics: list of metric dicts
    :return: dict of metric dicts with average
    """
    keys = side_metrics[0].keys()
    result = {}
    for key in keys:
        acum = sum(d[key] for d in side_metrics)
        result[key] = acum / len(side_metrics)
    return result


def productory(factors: torch.Tensor, dim=1) -> torch.Tensor:
    """
    Computes the matrix product of the sequence
    :param factors: * x n x n
    :param dim: acc
    :return: * x n x n: the result of multiplying all matrices on the dim dimension, according
    to the given order. The result will have one less dimension
    """
    m = factors.size(dim)
    acum = factors.select(dim=dim, index=0)
    for i in range(1, m):
        current = factors.select(dim=dim, index=i)
        acum = acum @ current
    return acum


def productory2(factors: torch.Tensor, dim=1) -> torch.Tensor:
    """
    Computes the matrix product of the sequence

    This is x1000 slower than 'productory'
    :param factors: * x n x n
    :param dim: acc
    :return: * x n x n: the result of multiplying all matrices on the dim dimension, according
    to the given order. The result will have one less dimension
    """
    b = factors.size(dim - 1)
    m = factors.size(dim)
    acum = []
    for i in range(b):
        mats = [t.squeeze(0) for t in factors.select(dim=dim-1, index=i).chunk(chunks=m)]  # list of m matrices of n x n
        prod = torch.chain_matmul(*mats)
        acum.append(prod)
    return torch.stack(acum, dim=0)


def trace(values: torch.Tensor, keepdim=False) -> torch.Tensor:
    """
    :param values: b x n x n
    :param keepdim:
    :return: b x 1 if keepdim == True else b
    """
    return torch.diagonal(values, dim1=-2, dim2=-1).sum(-1, keepdim=keepdim)


def get_run_id_with_epoch_name(run_id, epoch):
    run_number = ""
    if run_id[-1].isdigit() and run_id[-2] == "-":
        run_number = run_id[-1]
        run_id = run_id[:-2]
    run_id_with_epoch = f"{run_id}-ep{epoch}"
    return run_id_with_epoch + f"-{run_number}" if run_number else run_id_with_epoch
