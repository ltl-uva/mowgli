from __future__ import annotations
from collections import defaultdict
import logging
from typing import Callable, Optional, Tuple

import numpy as np

from mowgli.constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from mowgli.helpers import file2list
from mowgli.data import MowgliDataset, MowgliConcatDataset


logger = logging.getLogger(__name__)


class ParallelDataset(MowgliDataset):
    """
    Defines a parallel translation dataset, consisting of a source sentence and a corresponding target sentence. The data is read
    from disk and information utilized by the iterator is stored in memory. Note that the embedding lookup is not done here
    (this is computed on the fly).
    """
    def __init__(self, cfg, path, raw_data, src, trg, vocab, tokenizer, ):
        super().__init__()

        self.type = "parallel"
        self.raw_data = raw_data
        self.data = []
        self.tokenizer = tokenizer
        self.total_tokens = 0
        self.vocab = vocab
        share_vocab = True if "shared" in vocab.keys() else False
        self.src_key = "shared" if share_vocab else src
        self.trg_key = "shared" if share_vocab else trg
        self.trg_tag_enc = cfg["trg_tag_enc"]
        if self.trg_tag_enc:
            self.trg_tag = f"<2{trg}>"

        if not "train" in path:
            raw_sources, raw_targets = [], []

        for idx, sent in enumerate(raw_data):
            if sent[src] != "" and sent[trg] != "":
                src_sent, trg_sent = self.tokenizer(sent[src]), self.tokenizer(sent[trg])
                src_len, trg_len = len(src_sent), len(trg_sent)
                if src_len <= cfg["max_sent_length"] and trg_len <= cfg["max_sent_length"]:
                    self.total_tokens += src_len + trg_len
                    self.data.append(
                        {
                            "idx":          idx,
                            "src_len":      src_len,
                            "trg_len":      trg_len,
                            "src_lang":     src,
                            "trg_lang":     trg,
                            "pair":         f"{src}-{trg}",
                        }
                    )

                    if not "train" in path:
                        raw_sources.append(src_sent)
                        raw_targets.append(trg_sent)

        # DS: this is where the graph will be created, so that it's part of the dataset object.
        # Since this object is passed to `build_iterator()`, the function that handles building
        # the iterator, it is easy to include it in batches.
        self.graph = None

        # Store info for evaluation (only relevant for dev and test)
        if not "train" in path:
            self.src_lang = src
            self.trg_lang = trg
            self.direction = f"{src}-{trg}"
            self.src = raw_sources
            self.trg = raw_targets


    @classmethod
    def splits(cls, cfg, raw_data, tokenizer, vocab) -> Tuple[ParallelDataset, ParallelDataset, ParallelDataset]:
        """Returns train, validation and test splits by creating separate class instances for each."""
        def pairs_for_split(sources, targets):
            return [
                (src, trg)
                for src in sources
                for trg in targets
                if src != trg and
                cfg.get("centric") in [src, trg] or not cfg.get("centric")
            ]

        pairs = {}
        pairs["train"] = pairs_for_split(cfg["src"], cfg["trg"])
        # we might want to evaluate and test on different language pairs,
        # e.g. a subset of the training data or zero-shot directions
        pairs["valid"] = (
            pairs_for_split(cfg["valid_src"], cfg["valid_trg"])
            if cfg.get("valid_src") and cfg.get("valid_trg")
            else pairs["train"]
        )
        pairs["test"] = (
            pairs_for_split(cfg["test_src"], cfg["test_trg"])
            if cfg.get("test_src") and cfg.get("test_trg")
            else pairs["train"]
        )

        datasets = defaultdict(list)
        for split in pairs.keys():
            path = cfg.get(split+"_path")
            if not path: continue

            for src, trg in pairs[split]:
                data = cls(cfg=cfg, path=path, raw_data=raw_data[split], src=src, trg=trg, vocab=vocab, tokenizer=tokenizer)
                assert len(data) > 0
                logger.info(f"({src} -> {trg}) {cfg[split+'_path'].split('/')[-2]} size = {len(data)}")
                datasets[split].append(data)

        return (
            MowgliConcatDataset(datasets["train"]) if cfg.get("train_path") else None,
            datasets["valid"] if cfg.get("valid_path") else None,
            datasets["test"] if cfg.get("test_path") else None
        )
