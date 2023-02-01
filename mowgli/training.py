import argparse
import functools
import logging
import numpy as np
import operator
import os
import queue
import sys
import time
import wandb
from collections import Counter, defaultdict
from itertools import product
from typing import List

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.nn.parallel import DistributedDataParallel as DDP

from mowgli.builders import build_gradient_clipper, build_optimizer, build_scheduler
from mowgli.data import Batch, build_datasets, build_iterator, MultiparallelDataset
from mowgli.helpers import (
    log_data_info,
    log_cfg,
    load_checkpoint,
    make_logger,
    merge_iterators,
    set_seed,
    symlink_update,
    ConfigurationError,
    list2file,
    load_model_weights
)
from mowgli.loss import LabelSmoothedCrossEntropyLoss, SimilarityLoss
from mowgli.models import MowgliModel, build_model
from mowgli.prediction import validate_on_data, test

logger = logging.getLogger(__name__)

class TrainManager:
    """Manages training loop, validations, learning rate scheduling and early stopping."""
    def __init__(self, model: MowgliModel, cfg: dict, rank: int):
        # Model
        self.model = model
        self._log_parameters_list()
        self.model_type = cfg["model"].get("type", "universal")
        wandb.watch(self.model)

        # Logging and storing
        self.model_dir = cfg["training"]["model_dir"]
        assert os.path.exists(self.model_dir)
        self.logging_freq = cfg["training"].get("logging_freq", 100)
        self.valid_report_file = f"{self.model_dir}/validations.txt"

        # Data handling
        self.data_cfg = cfg["data"]
        self.level = cfg["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("invalid segmentation level. valid options: 'word', 'bpe', 'char'.")
        self.num_workers = cfg["data"].get("num_workers", 8)
        self.pin_memory = cfg["data"].get("pin_memory", True)
        self.shuffle = cfg["training"].get("shuffle", True)
        self.sample_temp = cfg["training"].get("sample_temperature", 1)
        assert (self.sample_temp == 1 or len(cfg["data"]["src"]) > 1 or len(cfg["data"]["trg"]) > 1
            ), "If sample temperature != 1, data should be multilingual"
        self.batch_size = cfg["training"]["batch_size"] # per-device => // self.n_gpu
        self.batch_multiplier = cfg["training"].get("batch_multiplier", 1)
        self.batch_type = cfg["training"].get("batch_type", "sentence")
        self.eval_batch_size = cfg["training"].get("eval_batch_size", self.batch_size) # per-device => // self.n_gpu
        self.eval_batch_type = cfg["training"].get("eval_batch_type", self.batch_type)
        self.pad_idx = self.model.pad_idx

        # Validation
        self.eval_beam_size = cfg["training"].get("eval_beam_size", 4)
        self.eval_beam_alpha = cfg["training"].get("eval_beam_alpha", 0.6)
        self.valid_before_train = cfg["training"].get("valid_before_train", False)
        self.validation_freq = cfg["training"].get("validation_freq", 1000)
        self.log_valid_sents = cfg["training"].get("print_valid_sents", [0, 1, 2])
        self.ckpt_queue = queue.Queue(maxsize=cfg["training"].get("keep_last_ckpts", 5))
        self.eval_metric = cfg["training"].get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf"]:
            raise ConfigurationError("Invalid setting for `eval_metric`, valid options: `bleu`, `chrf`.")
        self.early_stopping_patience = cfg["training"].get("early_stopping_patience", 10)
        self.max_output_length = cfg["training"].get("max_output_length", None)

        # Early stopping
        self.early_stopping_metric = cfg["training"].get("early_stopping_metric", "eval_metric")
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf"]:
                self.minimize_metric = False
            else:
                self.minimize_metric = True
        else:
            raise ConfigurationError("invalid setting for `early_stopping_metric`, valid options: `loss`, `ppl`, `eval_metric`.")

        # Training
        self.epoch_no = 0
        self.epochs = cfg["training"]["epochs"]
        self.max_update_steps = cfg["training"].get("max_update_steps", 1e6)
        self.label_smoothing = cfg["training"].get("label_smoothing", 0.0)
        self.learning_rate_min = cfg["training"].get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=cfg["training"])
        self.optimizer = build_optimizer(config=cfg["training"], parameters=model.parameters())
        self.model.set_loss_function(
            LabelSmoothedCrossEntropyLoss(pad_idx=self.pad_idx, smoothing=self.label_smoothing)
        )
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=cfg["training"],
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=cfg["model"]["decoder"]["hidden_size"]
        )
        self.weight_crossent = cfg["training"].get("weight_crossent_loss", 1.0)
        self.optimize_similarity_loss = True if cfg["training"].get("type_sim_loss", None) else False
        if self.optimize_similarity_loss:
            model.set_src_trg_context_saving(True)
            assert len(cfg["data"]["src"]) > 1 and cfg["data"].get("multiparallel", False)
            self.model.set_similarity_loss_function(
                SimilarityLoss(
                    self.pad_idx,
                    sim=cfg["training"].get("type_sim_loss"),
                    norm=cfg["training"].get("norm_sim_loss"),
                    layers=cfg["training"].get("layers_sim_loss", list(range(cfg["model"]["decoder"]["num_layers"])))
                )
            )
            self.weight_sim_loss = cfg["training"].get("weight_sim_loss", 1.0)
        self.normalization = cfg["training"].get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("invalid normalization option. valid options: `batch`, `tokens`, `none`.")

        # Testing
        self.bpe_type = cfg["testing"].get("bpe_type", "subword-nmt")
        self.sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in cfg["testing"].keys():
            self.sacrebleu["remove_whitespace"] = test_cfg["sacrebleu"].get("remove_whitespace", True)
            self.sacrebleu["tokenize"] = test_cfg["sacrebleu"].get("tokenize", "13a")

        # Handle device (CPU / GPU(s))
        self.rank = rank
        self.use_cuda = cfg["training"]["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.use_cuda else 0
        self.device = torch.device("cuda" if self.use_cuda else "cpu", index=self.rank)
        self.model.set_device(self.device)
        if self.use_cuda:
            logger.info("model to cuda...")
            self.model.to(self.device)
            if self.n_gpu > 1:
                logger.info("distribute model...")
                self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

        # Initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            stop=False,
            total_tokens=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric,
            valids_since_imp=0
        )

        if "load_model" in cfg["model"].keys():
            self.init_from_checkpoint(
                cfg["model"]["load_model"],
                reset_best_ckpt=cfg["training"].get("reset_best_ckpt", False),
                reset_scheduler=cfg["training"].get("reset_scheduler", False),
                reset_optimizer=cfg["training"].get("reset_optimizer", False),
                reset_steps=cfg["training"].get("reset_steps", False),
                reset_total_tokens=cfg["training"].get("reset_total_tokens", False)
            )


    def _save_checkpoint(self):
        "Save the model's current parameters and the training state to a checkpoint. The training state contains the "
        "total number of training steps, the total number of training tokens, the best checkpoint score and iteration "
        "so far, and optimizer and scheduler states."
        logger.info("saving new checkpoint.")
        model_path = f"{self.model_dir}/{self.stats.steps}.ckpt"
        model_state_dict = (
            self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict())
        state = {
            "steps": self.stats.steps,
            "total_tokens": self.stats.total_tokens,
            "best_ckpt_score": self.stats.best_ckpt_score,
            "best_ckpt_iteration": self.stats.best_ckpt_iter,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None}
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                logger.warning(f"tried to delete old checkpoint {to_delete} but does not exist.")
        self.ckpt_queue.put(model_path)
        best_path = f"{self.model_dir}/best.ckpt"
        try: symlink_update(f"{self.stats.steps}.ckpt", best_path) # create/modify symbolic link for best checkpoint
        except OSError: torch.save(state, best_path) # overwrite best.ckpt

    def init_from_checkpoint(
        self,
        path: str,                          # path to checkpoint
        reset_best_ckpt: bool = False,      # reset best checkpoint; use for e.g. new dev set or fine-tuning
        reset_scheduler: bool = False,      # reset the learning rate scheduler; do not use the one from checkpoint
        reset_optimizer: bool = False,      # reset the optimizer; do not use the one from checkpoint
        reset_steps: bool = False,          # reset steps; do not use steps from checkpoint
        reset_total_tokens: bool = False    # reset total tokens; do not use total tokens from checkpoint
    ):
        "Initialize the trainer from a given checkpoint file."
        logger.info(f"Loading model from {path}")
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        load_model_weights(self.model, model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.info("reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] and self.scheduler:
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            logger.info("reset scheduler.")

        if not reset_steps: self.stats.steps = model_checkpoint["steps"]
        else:               logger.info("reset update steps.")

        if not reset_total_tokens: self.stats.total_tokens = model_checkpoint["total_tokens"]
        else: logger.info("reset total tokens.")

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        # move parameters to gpu
        if self.use_cuda:
            self.model.cuda()

    def train_and_validate(self, train_data, valid_data) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\ttotal batch size (w. parallel & accumulation): %d",
            self.device, self.n_gpu, self.batch_multiplier, self.batch_size,
            (
                self.batch_size * self.batch_multiplier * self.n_gpu
                if self.n_gpu > 0
                else self.batch_size * self.batch_multiplier
            )
        )

        # validate before training, relevant when fine-tuning
        if self.valid_before_train:
            logger.info("Validate before starting training")
            self._validate(valid_data)

        for self.epoch_no in range(1, self.epochs):
            # Build iterator every epoch.
            logger.info("Building iterator...")
            train_iter = merge_iterators(
                [
                    build_iterator(
                        dataset             = data,
                        dataset_type        = data.type,
                        batch_size          = self.batch_size,
                        batch_type          = self.batch_type,
                        model_type          = self.model_type,
                        train               = True,
                        pad_idx             = self.pad_idx,
                        shuffle             = self.shuffle,
                        sample_temperature  = self.sample_temp,
                        distributed         = True if self.n_gpu > 1 else False,
                        num_workers         = self.num_workers,
                        pin_memory          = self.pin_memory,
                    )
                    for data in train_data
                ],
                batch_multiplier    = self.batch_multiplier,
                datasets            = train_data,
            )

            logger.info(f"Epoch {self.epoch_no}")

            if self.scheduler and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=self.epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.stats.total_tokens
            self.model.zero_grad()
            epoch_loss, batch_loss, batch_crossent, batch_similarity = 0, 0, 0, defaultdict(float)
            for i, batch in enumerate(iter(train_iter)):
                batch.to_device(self.device)
                # Get step losses
                step_losses = self._train_step(batch)

                # Increment total batch loss
                batch_loss += step_losses["norm_batch_loss"]
                batch_crossent += step_losses["norm_crossent_loss"]
                if step_losses["norm_sim_losses"] is not None:
                    for l in step_losses["norm_sim_losses"].keys():
                        batch_similarity[l] += step_losses["norm_sim_losses"][l]

                # Do update step
                if (i + 1) % self.batch_multiplier == 0:
                    # Clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        self.clip_grad_fun(params=self.model.parameters())

                    # Make gradient step
                    self.optimizer.step()

                    # Decay lr
                    if self.scheduler and self.scheduler_step_at == "step": self.scheduler.step()

                    # Reset gradients
                    self.model.zero_grad()

                    # Increment step counter
                    self.stats.steps += 1

                    # Gather to be logged data
                    logging_train = {
                        "epoch":                                                        self.epoch_no,
                        "update step":                                                  self.stats.steps,
                        "total seen tokens":                                            self.stats.total_tokens,
                        f"({batch.type}) train batch loss (norm)":                      batch_loss,
                        f"({batch.type}) train batch crossentropy loss (norm)":         batch_crossent,
                        **{f"({batch.type}) train similarity loss layer {l} (norm)":    sim for l, sim in batch_similarity.items()},
                        "batch type":                                                   batch.type,
                        "learning rate":                                                self.optimizer.param_groups[0]["lr"],
                    }
                    wandb.log(logging_train, commit=False)

                    # log learning progress
                    if self.stats.steps % self.logging_freq == 0:
                        elapsed = time.time() - start - total_valid_duration
                        elapsed_tokens = self.stats.total_tokens - start_tokens
                        tok_per_sec = (elapsed_tokens/elapsed)*self.n_gpu if self.n_gpu > 0 else elapsed_tokens/elapsed
                        logging_train["tokens per second"] = tok_per_sec

                        self._log(logging_train)

                        start = time.time()
                        total_valid_duration = 0
                        start_tokens = self.stats.total_tokens

                    # Only add complete loss of full mini-batch to epoch_loss
                    epoch_loss += batch_loss    # accumulate epoch_loss

                    # reset losses
                    batch_loss, batch_crossent, batch_similarity = 0, 0, defaultdict(float)

                    # validate on the entire validation set
                    if self.stats.steps % self.validation_freq == 0:
                        valid_duration = self._validate(valid_data)
                        total_valid_duration += valid_duration

                    if self.stats.steps >= self.max_update_steps:
                        self.stats.stop = True
                        self.stats.stop_reason = f"after {self.max_update_steps} updates"
                        logger.info(f"training ended {self.stats.stop_reason}")

                if self.stats.stop: break
            if self.stats.stop: break
            wandb.log({"train epoch loss": epoch_loss, "update step": self.stats.steps, "epoch": self.epoch_no})
            logger.info(f"epoch {self.epoch_no}: total training loss {round(epoch_loss, 2)}")

        else:
            logger.info(f"Training ended after {self.epoch_no} epochs.")

        logger.info("best validation result (greedy) at step {}: {} {}".format(
            self.stats.best_ckpt_iter, self.stats.best_ckpt_score, self.early_stopping_metric))


    def _train_step(self, batch: Batch) -> torch.Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return: loss for batch (sum)
        """
        # make normalizer for loss
        if   self.normalization == "batch":     normalizer = batch.nseqs
        elif self.normalization == "tokens":    normalizer = batch.ntokens
        elif self.normalization == "none":      normalizer = 1
        else: raise NotImplementedError("only normalize by `batch` or `tokens` or `none`")

        norm_batch_loss = 0

        # Reactivate training
        self.model.train()

        # Get losses through forward pass
        losses = self.model(batch)

        # Normalize crossentropy loss by `batch`, `tokens`, or `none` (1) and `batch_multiplier`
        norm_crossent_loss = losses["crossent"] / normalizer
        norm_crossent_loss = norm_crossent_loss / self.batch_multiplier
        weighted_crossent_loss = norm_crossent_loss * self.weight_crossent
        norm_batch_loss += norm_crossent_loss

        if self.optimize_similarity_loss and batch.type == "multiparallel":
            # Normalize similarity loss by `batch`, `tokens`, or `none` (1) and `batch_multiplier`
            n_layers = len(losses["similarity"])
            norm_sim_losses = {l: sim_loss / normalizer for l, sim_loss in losses["similarity"].items()}
            norm_sim_losses = {l: sim_loss / self.batch_multiplier for l, sim_loss in norm_sim_losses.items()}
            # Apply weights to similarity losses
            weighted_sim_losses = {l: sim_loss * self.weight_sim_loss for l, sim_loss in norm_sim_losses.items()}
            norm_batch_loss += sum(weighted_sim_losses.values())

        # Accumulate gradients
        norm_batch_loss.backward()

        self.stats.total_tokens += batch.ntokens

        return {
            "norm_batch_loss":      norm_batch_loss.item(),
            "norm_crossent_loss":   norm_crossent_loss.item(),
            "norm_sim_losses":      {
                                        l: sim_loss.item() for l, sim_loss in norm_sim_losses.items()
                                    }   if self.optimize_similarity_loss and batch.type == "multiparallel" else None,
        }


    def _validate(self, valid_data):
        valid_start_time = time.time()
        directions = [valid_set.direction for valid_set in valid_data]
        valid_scores,valid_losses,valid_ppls,cross_attns = [], [], [], {}
        if len(self.data_cfg["src"]) > 1:
            valid_combis = MultiparallelDataset.pairs_for_split(self.data_cfg, train=False)

        # loop over all translation directions
        for i, direction in enumerate(directions):
            results = validate_on_data(
                batch_size          = self.eval_batch_size,
                data                = valid_data[i],
                eval_metric         = self.eval_metric,
                level               = self.level,
                model               = self.model,
                save_cross_attn     = len(self.data_cfg["src"]) > 1,
                use_cuda            = self.use_cuda,
                max_output_length   = self.max_output_length,
                compute_loss        = True,
                beam_size           = self.eval_beam_size,
                beam_alpha          = self.eval_beam_alpha,
                batch_type          = self.eval_batch_type,
                postprocess         = True,           # always remove BPE for validation
                bpe_type            = self.bpe_type,     # "subword-nmt" or "sentencepiece"
                sacrebleu_opt       = self.sacrebleu,   # sacrebleu options
                n_gpu               = self.n_gpu,
                num_workers         = self.num_workers,
                rank                = self.rank,
            )
            valid_scores.append(results["score"])
            valid_losses.append(results["loss"].cpu())
            valid_ppls.append(results["ppl"].cpu())

            if len(self.data_cfg["src"]) > 1:
                cross_attns[direction] = results["cross_attn"]

            self._log_examples(
                direction       = direction,
                sources_raw     = [v for v in results["sources_raw"]],
                sources         = results["sources"],
                hypotheses_raw  = results["decoded"],
                hypotheses      = results["hyps"],
                references      = results["refs"],
            )

            # Store current validation outputs in file in `self.model_dir.`
            list2file(fn=f"{os.path.join(self.model_dir, 'translations')}/{self.stats.steps}.{direction}.hyps", l=results["hyps"])

            valid_duration = time.time() - valid_start_time

            logging_valid = {
                f"valid {self.eval_metric} {direction}":    results["score"],
                f"valid loss (sum) {direction}":            results["loss"],
                f"valid loss (normalized) {direction}":     results["loss_norm"],
                f"valid ppl (normalized) {direction}":      results["ppl"],
                "update step":                              self.stats.steps,
                "epoch":                                    self.epoch_no,
                "valid_duration":                           valid_duration,
            }
            self._log(logging_valid)

        # Calculate cross attention similarities (if multiple source languages)
        if len(self.data_cfg["src"]) > 1 and len(self.data_cfg.get("valid_src", self.data_cfg["src"])) > 1:
            for combi in valid_combis:
                total_tokens = 0
                [src1, src2], trg = combi
                cross_attn1, cross_attn2 = cross_attns[f"{src1}-{trg}"], cross_attns[f"{src2}-{trg}"]
                cross_attn_sims = []

                # Find multiparallel overlap for src1 and src2 into trg
                shared_idxs = set(cross_attn1.keys()).intersection(set(cross_attn2.keys()))

                # Get similarity loss for all sentences
                if self.optimize_similarity_loss:
                    for idx in shared_idxs:
                        total_tokens += cross_attn1[idx][0].shape[0]
                        sim_fn = (
                            self.model.module.similarity_loss_function
                            if self.n_gpu > 1 and self.use_cuda
                            else self.model.similarity_loss_function
                        )

                        cross_attn_sims.append(
                            sim_fn(
                                x1      = [c.unsqueeze(0) for c in cross_attn1[idx]],
                                x2      = [c.unsqueeze(0) for c in cross_attn2[idx]],
                                lang1   = src1,
                                lang2   = src2,
                            )
                        )

                    # Sum all losses together, normalize by number of sentences
                    sim_losses_sum = dict(functools.reduce(operator.add, map(Counter, cross_attn_sims)))
                    sim_losses_sum = {l: sim.item() / len(sim_losses_sum) for l, sim in sim_losses_sum.items()}

                    # Normalize
                    sim_losses_norm = {l: sim / total_tokens for l, sim in sim_losses_sum.items()}

                    logging_valid_sim = {
                        **{f"{src1}-{src2} similarity loss (sum) layer {l}":        sim for l, sim in sim_losses_sum.items()},
                        **{f"{src1}-{src2} similarity loss (normalized) layer {l}": sim for l, sim in sim_losses_norm.items()},
                        "update step":                                              self.stats.steps,
                        "epoch":                                                    self.epoch_no,
                    }
                    self._log(logging_valid_sim)

        # use mean values to report and determine determine early stopping
        valid_loss  = np.mean(np.array(valid_losses))
        valid_ppl   = np.mean(np.array(valid_ppls))
        valid_score = np.mean(np.array(valid_scores))

        # log average scores for multilingual models
        if len(directions) > 1:
            valid_duration = time.time() - valid_start_time
            logging_valid = {
                "valid loss average":                   valid_loss,
                f"valid {self.eval_metric} average":    valid_score,
                "valid ppl average":                    valid_ppl,
                "update step":                          self.stats.steps,
                "epoch":                                self.epoch_no,
                "valid duration":                       valid_duration,
            }
            self._log(logging_valid)

        if   self.early_stopping_metric == "loss": ckpt_score = valid_loss
        elif self.early_stopping_metric == "ppl":  ckpt_score = valid_ppl
        else:                                      ckpt_score = valid_score

        new_best = False
        if self.stats.is_best(ckpt_score):
            self.stats.valids_since_imp = 0
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info("new best validation result {}!".format(self.early_stopping_metric))
            if self.ckpt_queue.maxsize > 0:
                new_best = True
                if self.rank == 0: self._save_checkpoint()
        else:
            # check patience, maybe early stop if no improvement
            self.stats.valids_since_imp += 1
            logger.info(f"no improvement on validation set (for {self.stats.valids_since_imp} evaluations).")
            if self.stats.valids_since_imp >= self.early_stopping_patience:
                self.stats.stop = True
                self.stats.stop_reason = f"{self.stats.valids_since_imp} times no improvement on validation set"

        if self.scheduler and self.scheduler_step_at == "validation": self.scheduler.step(ckpt_score)
        self._add_report(valid_score, valid_loss, valid_ppl, new_best)

        return valid_duration

    def _add_report(self, valid_score: float, valid_ppl: float, valid_loss: float, new_best: bool):
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stats.stop = True
            self.stats.stop_reason = f"since minimum lr {self.learning_rate_min} was reached"

        if self.rank == 0:
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t"
                    "LR: {:.8f}\t{}\n".format(
                        self.stats.steps, valid_loss, valid_ppl, self.eval_metric,
                        valid_score, current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f"Total params: {n_params}")
        trainable_params = [n for (n, p) in self.model.named_parameters() if p.requires_grad]
        logger.debug("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log(self, log_dict: dict):
        """Logs the keys and values in `log_dict` to wandb and terminal."""
        wandb.log(log_dict)
        logger.info(
            "\n"+"\n".join(["{}: {}".format(i[0], round(i[1], 3) if isinstance(i[1], float) else i[1])
            for i in log_dict.items()])
        )

    def _log_examples(
        self,
        direction: str,                         # translation direction
        sources: List[str],                     # decoded sources
        hypotheses: List[str],                  # decoded hypotheses
        references: List[str],                  # decoded references
        sources_raw: List[List[str]] = None,    # raw sources
        hypotheses_raw: List[List[str]] = None, # raw hypotheses
        references_raw: List[List[str]] = None, # raw references
    ):
        "Log a the first `self.log_valid_sents` sentences from given examples."
        for p in self.log_valid_sents:
            if p >= len(sources): continue

            logger.info("Example #%d", p)

            if sources_raw is not None:    logger.debug(f"\t({direction}) Raw source:     {sources_raw[p]}")
            if references_raw is not None: logger.debug(f"\t({direction}) Raw reference:  {references_raw[p]}")
            if hypotheses_raw is not None: logger.debug(f"\t({direction}) Raw hypothesis: {hypotheses_raw[p]}")

            logger.info(f"\t({direction}) Source:     {sources[p]}")
            logger.info(f"\t({direction}) Reference:  {references[p]}")
            logger.info(f"\t({direction}) Hypothesis: {hypotheses[p]}")


    class TrainStatistics:
        def __init__(
            self,
            steps: int = 0,                     # global update step counter
            stop: bool = False,                 # stop training if this flag is True
            total_tokens: int = 0,              # number of total tokens seen so far
            best_ckpt_iter: int = 0,            # store iteration point of best ckpt
            best_ckpt_score: float = np.inf,    # initial values for best scores
            minimize_metric: bool = True,       # minimize or maximize score
            valids_since_imp: int = 0           # number of validations since improvement, used for early stopping
        ):
            ""
            self.steps = steps
            self.stop = stop
            self.stop_reason = None
            self.total_tokens = total_tokens
            self.best_ckpt_iter = best_ckpt_iter
            self.best_ckpt_score = best_ckpt_score
            self.minimize_metric = minimize_metric
            self.valids_since_imp = valids_since_imp

        def is_best(self, score):
            "Determines whether current checkpoint score is best score so far."
            return score < self.best_ckpt_score if self.minimize_metric else score > self.best_ckpt_score


def train(cfg: dict, rank: int):
    """Main training function. After training, also test on test data if given."""
    set_seed(seed=cfg["training"].get("random_seed", 42))

    make_logger(cfg["training"]["model_dir"], mode="train", rank=rank)

    if rank != 0 or cfg["training"].get("wandb_dryrun", False):
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project=cfg["name"], config=cfg)

    distributed=False
    # initialize torch.distributed if using multiple GPUs
    if cfg["training"]["use_cuda"] and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("initialize DistributedDataParallel...")
        distributed=True
        torch.distributed.init_process_group(backend='nccl')

    # Load datasets, vocabulary, and vocabulary keys
    train_data, valid_data, test_data, vocab, src_key, trg_key = build_datasets(cfg["data"], distributed=distributed)

    # build an encoder-decoder model
    model = build_model(cfg=cfg["model"], vocab=vocab, src_key=src_key, trg_key=trg_key)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model, cfg, rank)

    # log all entries of config
    log_cfg(cfg)

    log_data_info(train_data, valid_data, test_data, vocab, src_key, trg_key)

    logger.info(str(model))

    # train the model
    trainer.train_and_validate(train_data, valid_data)

    # predict with the best model on validation and test
    if rank == 0:
        ckpt = f"{cfg['training']['model_dir']}/{trainer.stats.best_ckpt_iter}.ckpt"
        output_name = f"ckpt_step_{trainer.stats.best_ckpt_iter}"
        output_path = os.path.join(cfg["training"]["model_dir"], output_name)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        data_to_test = {"test": test_data, "vocab": vocab, "src_key": src_key, "trg_key": trg_key}
        test(cfg=cfg, ckpt=ckpt, output_path=output_path, datasets=data_to_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Mowgli-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,help="training configuration file (yaml).")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank (for distributed training).")
    args = parser.parse_args()

    train(args.config, args.local_rank)
