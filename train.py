import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Adam, AdamW, SGD
from torch.optim.adagrad import Adagrad
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from gyrospd import config
from gyrospd import losses
from gyrospd.utils import set_seed, get_logging, get_inverted_triples
from gyrospd.runner import Runner
from gyrospd.models import *
from gyrospd.manifolds.metrics import MetricType


def config_parser(parser):
    # Data options
    parser.add_argument("--data", required=True, type=str, help="Name of data set folder")
    parser.add_argument("--run_id", required=True, type=str, help="Name of model/run to export")
    # Model
    parser.add_argument("--model", default="tgattnspd", type=str, help="Model type: tgspd, tgattnspd")
    parser.add_argument("--metric", default="riem", type=str, help=f"Metrics: {[t.value for t in list(MetricType)]}")
    parser.add_argument("--dims", default=4, type=int, help="Dimensions for the model.")
    parser.add_argument("--train_bias", dest='train_bias', action='store_true', default=False,
                        help="Whether to train scaling or not.")
    parser.add_argument("--use_hrh", default=1, type=int, help="Whether to use HRH or RHR in lhs op.")
    parser.add_argument("--inverse_tail", default=0, type=int, help="Whether to use t or t^-1 as tail")

    parser.add_argument("--loss", choices=["BCELoss", "HingeLoss"], default="BCELoss", help="Loss function")
    parser.add_argument("--hinge_margin", default=1, type=float, help="Margin for hinge loss function")
    parser.add_argument("--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer")
    parser.add_argument("--regularizer_weight", default=0, type=float, help="Regularization weight")
    parser.add_argument("--neg_sample_size", default=1, type=int, help="Negative sample size, -1 to not use")
    parser.add_argument("--double_neg", action="store_true", default=False,
                        help="Whether to negative sample both head and tail entities")

    # optim and config
    parser.add_argument("--optim", default="adam", type=str, help="Optimization method.")
    parser.add_argument("--amsgrad", action="store_true", default=False, help="Use AMS grad in Adam or AdamW")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Starting learning rate.")
    parser.add_argument("--reduce_factor", default=2, type=float, help="Factor to reduce lr on plateau.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--val_every", default=5, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--patience", default=50, type=int, help="Epochs of patience for scheduler and early stop.")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Eval batch size. Has impact only on memory")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--burnin", default=10, type=int, help="Number of initial epochs to train with reduce lr.")
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of update steps to accumulate before backward.")
    parser.add_argument("--subsample", default=-1, type=float, help="Subsamples a % of valid triples")

    # Others
    parser.add_argument("--dtype", default="single", type=str, choices=["single", "double"], help="Machine precision")
    parser.add_argument("--local_rank", type=int, help="Local process rank assigned by torch.distributed.launch")
    parser.add_argument("--n_procs", default=4, type=int, help="Number of process to create")
    parser.add_argument("--load_ckpt", default="", type=str, help="Load model from this file")
    parser.add_argument("--results_file", default="out/results.csv", type=str, help="Exports final results to this file")
    parser.add_argument("--save_epochs", default=10001, type=int, help="Exports every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--debug", dest='debug', action='store_true', default=False, help="Debug mode")


def load_ckpt(args, log):
    saved_data = torch.load(args.load_ckpt) if args.load_ckpt else {}
    args.init_epoch = 0
    args.current_iter = 1
    if saved_data:
        local_rank = args.local_rank
        if local_rank == 0:
            log.info(f"Loaded CKPT: {args.load_ckpt}, Args in ckpt: {saved_data['args']}")
        args = saved_data["args"]
        args.local_rank = local_rank
        args.init_epoch = saved_data["epochs"]
        args.current_iter += 1
    return args, saved_data


def get_model(args, saved_data=None):
    if args.model == "tgspd":
        model = TgSPDModel(args)
    elif args.model == "tgrotspd":
        model = TgSPDRotationModel(args)
    elif args.model == "tgrefspd":
        model = TgSPDReflectionModel(args)
    elif args.model == "tgattnspd":
        model = TgSPDAttnModel(args)
    else:
        raise ValueError(f"Unrecognized model argument: {args.model}")

    model.to(config.DEVICE)
    model = DistributedDataParallel(model, device_ids=None)
    if saved_data:
        model.load_state_dict(saved_data["model_state"])
    return model


def get_optimizer(model, args, saved_data=None):
    if args.optim == "sgd":
        optim = SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optim == "adamw":
        optim = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optim == "adagrad":
        optim = Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unkown --optim option: {args.optim}")
    if saved_data:
        optim.load_state_dict(saved_data["optimizer_state"])
    return optim


def get_scheduler(optimizer, args, saved_data=None):
    patience = round(args.patience / args.val_every)
    factor = 1 / float(args.reduce_factor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, mode="max")
    if saved_data:
        scheduler.load_state_dict(saved_data["scheduler_state"])
    return scheduler


def build_data_loader(split, batch_size, shuffle, args):
    """
    :param split: torch.LongTensor b x 3 with triples (h, r, t)
    :param batch_size: int
    :param shuffle: bool
    :param args:
    :return: torch DataLoader set up with distributed sampler
    """
    tensor_dataset = TensorDataset(split)
    sampler = DistributedSampler(tensor_dataset, num_replicas=args.n_procs, rank=args.local_rank, shuffle=shuffle)
    data_loader = DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=True, sampler=sampler)
    return data_loader


def load_training_data(args, log):
    data_path = config.PREP_PATH / f"{args.data}/{config.PREPROCESSED_FILE}"
    log.info(f"Loading data from {data_path}")
    data = torch.load(str(data_path))

    rel2id = data["rel2id"]
    num_relations = len(rel2id)

    train, valid, test = data["train"], data["valid"], data["test"]

    if args.debug:
        train = train[:10]
        valid = valid[:4]
        test = test[:4]
        args.batch_size = 40

    augmented = get_inverted_triples(train, num_relations)
    train = np.vstack((train, augmented))
    valid_proportion = args.subsample if args.subsample > 0 else 1
    valid = valid[:int(len(valid) * valid_proportion)]
    inverted_valid = get_inverted_triples(valid, num_relations)
    inverted_test = get_inverted_triples(test, num_relations)

    train, valid, inverted_valid, test, inverted_test = [torch.LongTensor(split) for split in
                                                         [train, valid, inverted_valid, test, inverted_test]]

    train_batch_size = args.batch_size // args.n_procs
    eval_batch_size = args.eval_batch_size // args.n_procs
    log.info(f"Batch size {train_batch_size} for {args.local_rank}/{args.n_procs} processes")

    train_loader = build_data_loader(train, train_batch_size, shuffle=True, args=args)
    lhs_valid_loader = build_data_loader(valid, eval_batch_size, shuffle=False, args=args)
    rhs_valid_loader = build_data_loader(inverted_valid, eval_batch_size, shuffle=False, args=args)
    lhs_test_loader = build_data_loader(test, eval_batch_size, shuffle=False, args=args)
    rhs_test_loader = build_data_loader(inverted_test, eval_batch_size, shuffle=False, args=args)

    valid_loaders = {"lhs": lhs_valid_loader, "rhs": rhs_valid_loader}
    test_loaders = {"lhs": lhs_test_loader, "rhs": rhs_test_loader}

    # add the inverted relations into the rel2id dict
    invrel2id = {f"INV_{rel_name}": rel_id + num_relations for rel_name, rel_id in rel2id.items()}
    rel2id = {**rel2id, **invrel2id}

    return train_loader, valid_loaders, test_loaders, data["filters"], data["ent2id"], rel2id


def main():
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    log = get_logging()
    torch.set_default_dtype(torch.float64 if args.dtype == "double" else torch.float32)

    args, saved_data = load_ckpt(args, log)

    torch.autograd.set_detect_anomaly(args.debug)

    # sets random seed
    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    set_seed(seed)

    if args.local_rank == 0:
        log.info(args)

    dist.init_process_group(backend=config.BACKEND, init_method='env://') # world_size=args.n_procs, rank=args.local_rank)

    # correct parameters due to distributed training. In case of loading ckpt, this value will be
    # ignored when we load the optimizer state dict
    args.learning_rate *= args.n_procs

    train_loader, valid_loaders, test_loaders, filters, ent2id, rel2id = load_training_data(args, log)

    args.num_entities = len(ent2id)
    args.num_relations = len(rel2id)    # already has inverted relations

    model = get_model(args, saved_data)
    optimizer = get_optimizer(model, args, saved_data)
    scheduler = get_scheduler(optimizer, args, saved_data)
    loss = getattr(losses, args.loss)(args)

    if args.local_rank == 0:
        log.info(f"GPU's available: {torch.cuda.device_count()}")
        n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
        log.info(f"Entities: {args.num_entities}, relations: {args.num_relations}, dims: {args.dims}, "
                 f"number of parameters: {n_params}")
        log.info(model)
        log.info(f"Triples train: {len(train_loader.dataset)}, valid lhs: {len(valid_loaders['lhs'].dataset)}, "
                 f"test lhs: {len(test_loaders['lhs'].dataset)}")
        if args.model in ("spd", "tgspd") and args.metric == "fone" and args.dims % 2 == 1:
            log.info("WARNING: SPD with Fone Metric and uneven number of dimensions can be unstable!!!")

    runner = Runner(model, optimizer, scheduler=scheduler, loss=loss, ent2id=ent2id, rel2id=rel2id, args=args,
                    train_loader=train_loader, valid_loaders=valid_loaders, test_loaders=test_loaders, filters=filters)
    runner.run()
    log.info("Done!")


if __name__ == "__main__":
    main()
