
import copy
import time
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from tensorboardX import SummaryWriter
from gyrospd import config
from gyrospd.utils import get_logging, write_results_to_file, compute_metrics, avg_side_metrics, \
    get_run_id_with_epoch_name


class Runner(object):
    def __init__(self, model, optimizer, scheduler, loss, ent2id, rel2id, train_loader, valid_loaders, test_loaders,
                 filters, args):
        self.ddp_model = model
        self.model = model.module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.train_loader = train_loader
        self.valid_loader = valid_loaders
        self.test_loader = test_loaders
        self.filters = filters
        self.loss = loss
        self.args = args
        self.log = get_logging()
        self.is_main_process = args.local_rank == 0
        if self.is_main_process:
            self.writer = SummaryWriter(config.TENSORBOARD_PATH / args.run_id)

    def run(self):
        best_mrr, best_epoch, current_epoch = -1, -1, -1
        best_model_state = copy.deepcopy(self.ddp_model.state_dict())
        for epoch in range(self.args.init_epoch + 1, self.args.epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)      # sets epoch for shuffling
            self.set_burnin_lr(epoch)
            start = time.perf_counter()
            train_loss = self.train_epoch(self.train_loader, epoch)
            exec_time = time.perf_counter() - start

            if self.is_main_process:
                self.log.info(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')
                self.write_tensorboard(train_loss, epoch)

            if epoch % self.args.save_epochs == 0 and self.is_main_process:
                self.save_model(epoch)

            if epoch % self.args.val_every == 0:
                start = time.perf_counter()
                metrics, _ = self.evaluate(self.valid_loader)
                eval_time = time.perf_counter() - start
                mrr = metrics["MRR"]
                if self.is_main_process:
                    for key in ["MRR", "MR", "HR@1", "HR@3", "HR@10"]:
                        self.writer.add_scalar(f"val/{key}", metrics[key], epoch)

                self.log.info(f"RANK {self.args.local_rank}: Results ep {epoch}: time: {int(eval_time)} s, "
                              f"tr loss: {train_loss:.1f}, MRR: {mrr * 100:.2f}, MR: {metrics['MR']:.0f}, "
                              f"HR@10: {metrics['HR@10'] * 100:.2f}")

                self.scheduler.step(mrr)

                if mrr > best_mrr:
                    if self.is_main_process:
                        self.log.info(f"Best val MRR: {mrr * 100:.3f}, at epoch {epoch}")
                    best_mrr = mrr
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.ddp_model.state_dict())

                # early stopping
                if epoch - best_epoch >= self.args.patience * 10:
                    self.log.info(f"RANK {self.args.local_rank}: Early stopping at epoch {epoch}!!!")
                    break

        # It runs evaluation on test using the best model so far
        self.log.info(f"RANK {self.args.local_rank}: Final evaluation: loading best model from epoch {best_epoch}")
        self.ddp_model.load_state_dict(best_model_state)

        test_metrics, ranks_and_rels = self.evaluate(self.test_loader)

        if self.is_main_process:
            self.export_results(test_metrics, best_epoch)
            self.print_results_per_relation(ranks_and_rels)
            self.log.info(f"Final Results: MRR: {test_metrics['MRR'] * 100:.2f}, MR: {test_metrics['MR']:.0f}, "
                          f"HR@1: {test_metrics['HR@1'] * 100:.2f}, HR@3: {test_metrics['HR@3'] * 100:.2f}, "
                          f"HR@10: {test_metrics['HR@10'] * 100:.2f}")

            self.writer.close()

    def train_epoch(self, train_split, epoch_num):
        tr_loss = 0.0
        avg_grad_norm = 0.0
        self.ddp_model.train()
        self.ddp_model.zero_grad()
        self.optimizer.zero_grad()
        dist_to_pos_acum, dist_to_neg_acum = 0, 0
        n_pos, n_neg = 0, 0
        rels, attn_weights = [], []     # only for attn model

        for step, triples in enumerate(train_split):
            loss, dist_to_pos, dist_to_neg = self.loss.calculate_loss(self.model, triples[0].to(config.DEVICE))
            loss = loss / self.args.grad_accum_steps
            loss.backward()

            # update
            if (step + 1) % self.args.grad_accum_steps == 0:
                grad_norm = clip_grad_norm_(self.ddp_model.parameters(), self.args.max_grad_norm)
                avg_grad_norm += grad_norm.item()
                self.optimizer.step()
                self.ddp_model.zero_grad()
                self.optimizer.zero_grad()

            # stats
            tr_loss += loss.item()
            dist_to_pos_acum += dist_to_pos.sum().item()
            dist_to_neg_acum += dist_to_neg.sum().item()
            n_pos += len(dist_to_pos)
            n_neg += len(dist_to_neg)
            if self.is_main_process and hasattr(self.model, "attn_weights"):
                rels.append(triples[0][:, 1].unsqueeze(-1))                         # b x 1
                attn_weights.append(self.model.attn_weights[:len(triples[0])])      # b x 2

        if self.is_main_process:
            self.writer.add_scalar("avg_norm/grad", avg_grad_norm / len(train_split), epoch_num)
            dists = {"pos": dist_to_pos_acum / n_pos, "neg": dist_to_neg_acum / n_neg}
            self.writer.add_scalars("avg_norm/dist", dists, epoch_num)
            if len(attn_weights):
                self.write_attn_weights(rels, attn_weights, epoch_num)
        return tr_loss / len(train_split)

    def evaluate(self, eval_loaders_dict):
        """
        :param eval_loaders_dict: dict of evaluation loaders: {"lhs": DataLoader, "rhs": DataLoader}
        :return: loss
        :return: metrics: dict with keys: ["MRR", "MR", "HR@1", "HR@3", "HR@10"], and float values
        :return: ranks_and_rels: list with tensors b x 2, where idx 0 is ranking and idx 1 is rel_id
        """
        self.ddp_model.eval()
        side_metrics = []
        ranks_and_rels = []

        for side in ["lhs", "rhs"]:
            if self.is_main_process:
                self.log.info(f"Evaluating {side}")
            loader = eval_loaders_dict[side]
            filters = self.filters[side]
            ranks, relation_ids = [], []

            for j, batch in enumerate(loader):
                if self.is_main_process:
                    self.log.info(f"Batch {j + 1}/{len(loader)}: len: {len(batch[0])}")

                with torch.no_grad():
                    all_scores, target_scores = self.model.evaluate(batch[0])

                    # set filtered and true scores to -1e6 to be ignored
                    for i, triple in enumerate(batch[0]):
                        to_filter_out = filters[(triple[0].item(), triple[1].item())]   # uses (h, r) as dict key
                        all_scores[i, to_filter_out] = -1e6

                    this_rank = torch.sum((all_scores >= target_scores).float(), dim=1, keepdim=True) + 1    # b x 1
                    ranks.append(this_rank)
                    relation_ids.append(batch[0][:, 1].unsqueeze(-1).float())

            local_rank = torch.cat(ranks, dim=0)                # len_loader_for_this_process x 1
            local_rel_ids = torch.cat(relation_ids, dim=0)      # len_loader_for_this_process x 1
            local_rank_and_rels = torch.cat([local_rank, local_rel_ids], dim=-1).float()     # len x 2

            gathered_ranks_and_rels = [torch.ones_like(local_rank_and_rels).float() for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=gathered_ranks_and_rels, tensor=local_rank_and_rels)
            global_rank_and_rels = torch.cat(gathered_ranks_and_rels, dim=0)    # b x 2
            side_metrics.append(compute_metrics(global_rank_and_rels[:, 0]))
            ranks_and_rels.append(global_rank_and_rels)

        metrics = avg_side_metrics(side_metrics)
        return metrics, ranks_and_rels

    def save_model(self, epoch):
        save_path = config.CKPT_PATH / self.get_run_id_with_epoch_name(self.args.run_id, epoch)
        self.log.info(f"Saving model checkpoint to {save_path}")
        torch.save({
            "model_state": self.ddp_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epochs": epoch,
            "args": self.args
        }, save_path)
        return save_path

    def write_tensorboard(self, train_loss, epoch):
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/lr", self.get_lr(), epoch)
        self.writer.add_scalar("avg_norm/entities", self.model.entity_norms().mean().item(), epoch)
        self.writer.add_scalar("avg_norm/relations", self.model.relation_norms().mean().item(), epoch)
        self.writer.add_scalar("avg_norm/rel_transf", self.model.relation_transform_norms().mean().item(), epoch)
        biases = {"head": self.model.bias_head.weight.detach().mean().item(),
                  "tail": self.model.bias_tail.weight.detach().mean().item()}
        self.writer.add_scalars("avg_norm/biases", biases, epoch)

    def write_attn_weights(self, rels, attn_weights, epoch):
        """
        :param rels: list of b x 1 tensors with rel ids
        :param attn_weights: list of b x 2 tensors with rot, ref attn weights
        :param epoch:
        """
        rels = torch.cat(rels, dim=0).squeeze(-1)       # all
        attn_weights = torch.cat(attn_weights, dim=0)   # all x 2
        for rel_name, rel_id in self.rel2id.items():
            rel_index = rels == rel_id
            if not torch.any(rel_index):
                continue
            this_rot_weights = attn_weights[:, 0][rel_index]
            this_ref_weights = attn_weights[:, 1][rel_index]
            scalars = {"rot": this_rot_weights.mean().item(), "ref": this_ref_weights.mean().item()}
            self.writer.add_scalars(f"weights_att/{rel_name}", scalars, epoch)

    def print_results_per_relation(self, ranks_and_rels):
        """:param ranks_and_rels: list with tensors b x 2, where idx 0 is ranking and idx 1 is rel_id"""
        to_print = []
        for rel_name, rel_id in self.rel2id.items():
            for this_ranks_and_rels in ranks_and_rels:
                filtered_rels = this_ranks_and_rels[this_ranks_and_rels[:, 1] == rel_id]
                if len(filtered_rels) == 0:     # one side of rank_and_rels will always give an empty tensor
                    continue

                metrics = compute_metrics(filtered_rels[:, 0])
                to_print.append(f"Relation '{rel_name}' ({len(filtered_rels)}): MRR: {metrics['MRR'] * 100:.2f}, MR: {metrics['MR']:.0f}, "
                                f"HR@10: {metrics['HR@10'] * 100:.2f}")

        to_print = "\n".join(to_print)
        self.log.info(f"Results per relation:\n{to_print}")

    def set_burnin_lr(self, epoch):
        """Modifies lr if epoch is less than burn-in epochs"""
        if self.args.burnin < 1:
            return
        if epoch == 1:
            self.set_lr(self.get_lr() / config.BURNIN_FACTOR)
        if epoch == self.args.burnin:
            self.set_lr(self.get_lr() * config.BURNIN_FACTOR)

    def set_lr(self, value):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = value

    def get_lr(self):
        """:return current learning rate as a float"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def check_points_in_manifold(self):
        """it checks that all the points are in the manifold"""
        all_points_ok, outside_point, reason = self.model.check_all_points()
        if not all_points_ok:
            raise AssertionError(f"Point outside manifold. Reason: {reason}\n{outside_point}")

    def export_results(self, metrics, best_epoch):
        """
        :param metrics: dict with keys: ["MRR", "MR", "HR@1", "HR@3", "HR@10"], and float values
        :param best_epoch: int
        :return:
        """
        model = self.args.model
        dims = self.args.dims

        if model in {"spd", "tgspd", "tgrotspd", "tgrefspd", "tgattnspd"}:
            dims = dims * (dims + 1) / 2
        if model in {"upper", "bounded", "dual", "tgupper"}:
            dims = dims * (dims + 1)
        model += "-" + self.args.metric

        result_data = {"data": self.args.data, "dims": dims, "manifold": model,
                       "run_id": get_run_id_with_epoch_name(self.args.run_id, best_epoch)}
        for k, v in metrics.items():
            factor = 100 if k != "MR" else 1
            result_data[k] = v * factor

        write_results_to_file(self.args.results_file, result_data)
