import gyrospd.utils as utils
import gyrospd.config as config
import collections
import numpy as np
import torch

log = utils.get_logging()


def get_idx(path):
    """
    Map entities and relations to unique ids.

    :param path: path to directory with raw dataset files (tab-separated train/valid/test triples)
    :return: ent2id: Dictionary mapping raw entities to unique ids
    :return: rel2id: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        with open(path / split, "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split()
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    ent2id = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2id = {x: i for (i, x) in enumerate(sorted(relations))}
    return ent2id, rel2id


def to_np_array(dataset_file, ent2id, rel2id):
    """
    Map raw dataset file to numpy array with unique ids.

    :param dataset_file: Path to file containing raw triples in a split
    :param ent2id: Dictionary mapping raw entities to unique ids
    :param rel2id: Dictionary mapping raw relations to unique ids
    :return: Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split()
            try:
                examples.append([ent2id[lhs], rel2id[rel], ent2id[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")


def get_filters(examples, n_relations):
    """
    Create filtering lists for evaluation.

    :param examples: Numpy array of size n_examples x 3 containing KG triples
    :param n_relations: Int indicating the total number of relations in the KG
    :return: lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for
    left-hand-side prediction
    :return: rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for
    right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        lhs_filters[(lhs, rel)].add(rhs)
        rhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final, rhs_final = {}, {}
    for k, v in lhs_filters.items():
        lhs_final[k] = torch.LongTensor(sorted(list(v)))
    for k, v in rhs_filters.items():
        rhs_final[k] = torch.LongTensor(sorted(list(v)))
    return lhs_final, rhs_final


def process_dataset(path):
    """
    :param path: Path to dataset directory
    :return:
    """
    ent2id, rel2id = get_idx(dataset_path)
    train, valid, test = [to_np_array(path / split, ent2id, rel2id) for split in ("train", "valid", "test")]

    all_examples = np.concatenate((train, valid, test), axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2id))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return train, valid, test, filters, ent2id, rel2id


if __name__ == "__main__":
    data_path = config.PREP_PATH
    log.info(f"Data Path: {data_path}")
    for dataset_path in data_path.iterdir():
        log.info(f"Processing: {dataset_path}")
        train, valid, test, filters, ent2id, rel2id = process_dataset(dataset_path)

        log.info(f"Entities: {len(ent2id)}, relations: {len(rel2id)}, "
                 f"Train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

        torch.save({
            "train": train,
            "valid": valid,
            "test": test,
            "filters": filters,
            "ent2id": ent2id,
            "rel2id": rel2id
        }, dataset_path / config.PREPROCESSED_FILE)
