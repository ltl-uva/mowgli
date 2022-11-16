from collections import defaultdict
from itertools import product, repeat
import logging
from typing import Callable, Optional, Tuple

import numpy as np

from mowgli.constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from mowgli.data import MowgliDataset, MowgliConcatDataset, RawTextDataset
from mowgli.helpers import file2list


logger = logging.getLogger(__name__)


class MultiparallelDataset(MowgliDataset):
    """
    Defines a multiparallel translation dataset, consisting of two meaning equivalent source sentences and a corresponding
    target sentence. The data is read from disk and information utilized by the iterator is stored in memory. Note that the
    embedding lookup is not done here (this is computed on the fly).
    """

    def __init__(self, cfg: dict, path: str, raw_data: RawTextDataset, src: list, trg: str, vocab: dict, tokenizer):
        super().__init__()

        self.type = "multiparallel"
        self.raw_data = raw_data
        self.data = []
        self.total_tokens = 0
        self.tokenizer = tokenizer
        self.vocab = vocab
        share_vocab = True if "shared" in vocab.keys() else False
        src1, src2 = src[0], src[1]
        max_len = cfg["max_sent_length"]
        self.src1_key = "shared" if share_vocab else src1
        self.src2_key = "shared" if share_vocab else src2
        self.trg_key = "shared" if share_vocab else trg
        self.trg_tag_enc = cfg["trg_tag_enc"]
        if self.trg_tag_enc:
            self.trg_tag = f"<2{trg}>"


        for idx, sent in enumerate(raw_data):
            if sent[src2] != "" and sent[src1] != "" and sent[trg] != "":
                src1_sent, src2_sent, trg_sent = self.tokenizer(sent[src1]), self.tokenizer(sent[src2]), self.tokenizer(sent[trg])
                src1_len, src2_len, trg_len = len(src1_sent), len(src2_sent), len(trg_sent)
                if src1_len <= max_len and src2_len <= max_len and trg_len <= max_len:
                    self.total_tokens += src1_len + src2_len + trg_len
                    self.data.append(
                        {
                            "idx":          idx,
                            "src1_len":     src1_len,
                            "src2_len":     src2_len,
                            "src_len":      max(src1_len, src2_len),
                            "trg_len":      trg_len,
                            "src1_lang":    src1,
                            "src2_lang":    src2,
                            "trg_lang":     trg,
                        }
                    )


    @classmethod
    def splits(cls, cfg, raw_data, tokenizer, vocab):
        """Returns train split by creating separate class instance."""
        assert cfg.get("train_path")

        datasets = defaultdict(list)

        for src, trg in cls.pairs_for_split(cfg):

            data = cls(cfg=cfg, path=cfg["train_path"], raw_data=raw_data["train"], src=src, trg=trg, vocab=vocab, tokenizer=tokenizer)
            assert len(data) > 0
            logger.info(f"([{src[0]}, {src[1]}] -> {trg}) train size = {len(data)}")
            datasets["train"].append(data)

        return MowgliConcatDataset(datasets["train"])


    def pairs_for_split(cfg, train=True):
        """Returns multiparallel (src1,src2)->trg pairs."""
        centric = cfg.get("centric")
        pairs, seen = [],[]

        # Create multiparallel {src1,src2}->trg combinations.
        # Use valid_src and valid_trg if available, else use train src trg
        for (src1, src2, trg) in product(
            *list(repeat(cfg["src"] if train else cfg.get("valid_src", cfg["src"]), 2)),
                         cfg["trg"] if train else cfg.get("valid_trg", cfg["trg"])
        ):
            srcs = [src1, src2]
            if (
                src1 == src2 or sorted(srcs) in seen or     # skip if duplicate src or already seen
                trg in [src1, src2] or                      # skip copy tasks
                (centric and centric not in trg)            # skip if centric language not in trg
            ):
                continue
            seen.append(sorted(srcs))
            pairs.append([sorted(srcs), trg])

        return pairs
