from collections import Counter
import copy
import math
import io
import sys
import random
import logging
import torch

from mowgli.constants import (
    UNK_TOKEN,
    PAD_TOKEN,
    DEFAULT_UNK_ID
)
from mowgli.data import (
    Batch,
    MultiparallelBatch,
    BatchSampler,
    DistributedBatchSampler,
    ParallelDataset,
    RawTextDataset,
    TemperatureSampler,
    MultiparallelDataset,
    Vocabulary,
)
from mowgli.helpers import file2list, flatten

logger = logging.getLogger(__name__)


def build_vocab(
    train_path,
    src,
    trg,
    shared_vocab,
    tokenizer,
    reduce_size,
    trg_tag,
    max_size: int,
    min_freq: int,
    vocab_file: str = None
) -> dict:
    """
    Builds vocabulary.
    """
    languages = src + trg

    tags = ["<2{}>".format(l) for l in trg] if trg_tag else []

    if vocab_file is not None:
        # load it from file
        logger.info("load vocab from file...")
        keys = ["shared"] if shared_vocab else languages
        vocab = {k: Vocabulary(file=vocab_file+"."+k) for k in keys}

    else:
        # create newly
        logger.info("create new vocab...")

        def filter_min(counter: Counter, min_freq: int):
            """ Filter counter by min frequency """
            filtered_counter = Counter(
                {t: c for t, c in counter.items() if c >= min_freq}
            )
            return filtered_counter

        def sort_and_cut(counter: Counter, limit: int):
            """ Cut counter to most frequent,
            sorted numerically and alphabetically"""
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(
                counter.items(), key=lambda tup: tup[0]
            )
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
            return vocab_tokens

        data = {l: file2list(f"{train_path}.{l}", reduce_size=reduce_size) for l in languages}

        if shared_vocab:
            data = {"shared": flatten(data.values())}

        vocab = {}
        for k in data.keys():
            tokens = []
            for s in data[k]:
                tokens.extend(tokenizer(s))

            counter = Counter(tokens)
            if min_freq > -1:
                counter = filter_min(counter, min_freq)
            vocab_tokens = sort_and_cut(counter, max_size)
            assert len(vocab_tokens) <= max_size

            vocab_k = Vocabulary(tokens=vocab_tokens, specials=tags)
            if trg_tag:
                vocab_k.specials += tags

            assert len(vocab_k) <= max_size + len(vocab_k.specials)
            assert vocab_k.i2s[DEFAULT_UNK_ID()] == UNK_TOKEN

            vocab[k] = vocab_k

        # check for all except for UNK token whether they are OOVs
        for k in vocab.keys():
            for s in vocab[k].specials[1:]:
                assert not vocab[k].is_unk(s)

        # save vocab
        fn = "/".join(train_path.split("/")[:-2])+"/vocab."
        for k in vocab.keys():
            vocab[k].to_file(fn+k)

    return vocab

def build_datasets(cfg: dict, datasets: list = ["train", "valid", "test"], distributed: bool = False):
    assert (
        cfg.get("train_path") or
        cfg.get("valid_path") or
        cfg.get("test_path") or
        cfg.get("name") and cfg.get("data_path")
    ), "Specify train valid test paths or dataset name and path."

    src_max_size = cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = cfg.get("src_voc_min_freq", 1)
    trg_max_size = cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = cfg.get("trg_voc_min_freq", 1)
    assert isinstance(cfg["src"], list)
    assert isinstance(cfg["trg"], list)

    def tok_char(s): return list(s)
    def tok_word(s): return s.split()
    tok_fun = tok_char if cfg["level"] == "char" else tok_word

    assert cfg.get("train_path") or cfg.get("vocab")

    vocab = build_vocab(
        train_path      = cfg.get("train_path"),
        src             = cfg["src"],
        trg             = cfg["trg"],
        shared_vocab    = cfg["share_vocab"],
        tokenizer       = tok_fun,
        reduce_size     = cfg.get("reduce_size", None),
        trg_tag         = True if cfg.get("trg_tag_enc", False) else False,
        max_size        = max(src_max_size, trg_max_size),
        min_freq        = min(src_min_freq, trg_min_freq),
        vocab_file      = cfg.get("vocab"),
    )

    train_data, valid_data, test_data = None, None, None
    logger.info("loading data...")

    assert (
        ("train" in datasets and cfg.get("train_path")) or
        ("valid" in datasets and cfg.get("valid_path")) or
        ("test"  in datasets and cfg.get("test_path"))
    )

    # Load raw dataset in memory
    raw_data = RawTextDataset.splits(cfg)

    # Load parallel training, validation and test datasets
    assert cfg.get("parallel", True) or cfg.get("multiparallel", False)
    if cfg.get("parallel", True):
        train_data = []
        parallel_train_data, valid_data, test_data = ParallelDataset.splits(
            cfg         = cfg,
            raw_data    = raw_data,
            tokenizer   = tok_fun,
            vocab       = vocab,
        )
        train_data.append(parallel_train_data)
        # Mix parallel and multiparallel training data
        if cfg.get("multiparallel", False):
            multiparallel_train_data = MultiparallelDataset.splits(cfg=cfg, raw_data=raw_data, tokenizer=tok_fun, vocab=vocab)
            train_data.append(multiparallel_train_data)

    # Load multiparallel training dataset, and parallel validation and test datasets
    elif cfg.get("multiparallel", False):
        train_data = []
        multiparallel_train_data = MultiparallelDataset.splits(cfg=cfg, raw_data=raw_data, tokenizer=tok_fun, vocab=vocab)
        train_data.append(multiparallel_train_data)

        cfg_parallel = copy.copy(cfg)
        cfg_parallel["train_path"] = None
        _, valid_data, test_data = ParallelDataset.splits(cfg=cfg_parallel, raw_data=raw_data, tokenizer=tok_fun, vocab=vocab)

    logger.info("data loaded.")

    src_key = ["shared"] if cfg["share_vocab"] else cfg["src"]
    trg_key = ["shared"] if cfg["share_vocab"] else cfg["trg"]

    return train_data, valid_data, test_data, vocab, src_key, trg_key


def build_iterator(
    dataset,
    batch_size: int,
    batch_type: str,
    train: bool,
    pad_idx: int,
    distributed: bool = False,
    shuffle: bool = False,
    model_type: str = "universal",
    sample_temperature: int = 1,
    dataset_type: str = "parallel",
    num_workers: int = 8,
    pin_memory: bool = True
) -> torch.utils.data.dataloader.DataLoader:
    """
    Builds a torch.utils.data.DataLoader object for iterating through dataset.

    :param dataset: dataset for which to create an iterator
    :param batch_type: batch size is calculated using #sentences or #tokens
    :param batch_size: batch size
    :param train: whether the iterator is used for training
    :param world_size: number of parallel processes, typically the number
        of gpus
    :param pad_idx: padding token index
    """
    assert batch_type in ("token", "sentence"), "`batch_type` type {} is not implemented.".format(batch_type)

    # when we use multiple processes, create larger virtual batches and use parts of those on each device.
    if train and distributed:
        if not torch.distributed.is_available():
            raise RuntimeError("Requires torch.distributed package to be available.")
        batch_size = batch_size * torch.distributed.get_world_size()

    def parallel_collator(batch):
        """Collate lists of parallel samples into batches"""
        src = [torch.LongTensor(x["src"]) for x in batch]
        trg = [torch.LongTensor(x["trg"]) for x in batch]

        return Batch(
            src         = torch.nn.utils.rnn.pad_sequence(src, padding_value=pad_idx, batch_first=True),
            trg         = torch.nn.utils.rnn.pad_sequence(trg, padding_value=pad_idx, batch_first=True),
            src_length  = torch.LongTensor([x["src_len"] for x in batch]),
            trg_length  = torch.LongTensor([x["trg_len"] for x in batch]),
            pad         = pad_idx,
            idxs        = [x["idx"] for x in batch],
            src_lang    = batch[0]["src_lang"],
            trg_lang    = batch[0]["trg_lang"]
        )

    def multiparallel_collator(batch):
        src1 = [torch.LongTensor(x["src1"]) for x in batch]
        src2 = [torch.LongTensor(x["src2"]) for x in batch]
        trg =  [torch.LongTensor(x["trg"])  for x in batch]

        return MultiparallelBatch(
            src1        = torch.nn.utils.rnn.pad_sequence(src1, padding_value=pad_idx, batch_first=True),
            src2        = torch.nn.utils.rnn.pad_sequence(src2, padding_value=pad_idx, batch_first=True),
            trg         = torch.nn.utils.rnn.pad_sequence(trg,  padding_value=pad_idx, batch_first=True),
            src1_length = torch.LongTensor([x["src1_len"] for x in batch]),
            src2_length = torch.LongTensor([x["src2_len"] for x in batch]),
            trg_length  = torch.LongTensor([x["trg_len"]  for x in batch]),
            pad         = pad_idx,
            src1_lang   = batch[0]["src1_lang"],
            src2_lang   = batch[0]["src2_lang"],
            trg_lang    = batch[0]["trg_lang"]
        )

    if dataset_type == "parallel":
        sort_key = lambda x: (x["src_len"], x["trg_len"])
        collator = parallel_collator
        batch_size_fn = parallel_ntokens if batch_type == "token" else lambda new, count, sofar: count

    elif dataset_type == "multiparallel":
        collator = multiparallel_collator
        # restrict languages for modular models
        if model_type == "modular":
            sort_key = lambda x: (x["src1_lang"], x["src2_lang"], x["trg_lang"], x["src_len"], x["trg_len"])
            batch_size_fn = multiparallel_ntokens_limit_langs if batch_type == "token" else multiparallel_nsents_limit_langs

        elif model_type == "universal":
            sort_key = lambda x: (x["src_len"], x["trg_len"])
            batch_size_fn = multiparallel_ntokens if batch_type == "token" else multiparallel_nsents

    else:
        raise NotImplementedError("Dataset type `{}` if not implemented.".format(dataset_type))

    # Temperature based data sampling for multilingual datasets
    if train and sample_temperature != 1:
        logger.info("Resample with temperature {}".format(sample_temperature))
        temp_sampler = TemperatureSampler(dataset, temp=sample_temperature)
        dataset = temp_sampler.sample()

    batch_sampler = BatchSampler(
        dataset         = dataset,
        batch_size      = batch_size,
        sort_key        = sort_key,
        batch_size_fn   = batch_size_fn,
        train           = train,
        shuffle         = shuffle,
    )

    # wrap with distributed sampler if we use multiple GPUs if train
    if train and distributed:
        batch_sampler = DistributedBatchSampler(batch_sampler)

    # Use our batch sampler to define a torch DataLoader
    return torch.utils.data.DataLoader(
        dataset         = dataset,
        batch_sampler   = batch_sampler,
        collate_fn      = collator,
        pin_memory      = pin_memory,
        num_workers     = num_workers,
    )

global max_src_in_batch, max_trg_in_batch
def parallel_ntokens(new: tuple, count: int, sofar: int) -> int:
    global max_src_in_batch, max_trg_in_batch
    new_data = new[1]
    if count == 1:
        max_src_in_batch = 0
        max_trg_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, new_data["src_len"])
    max_trg_in_batch = max(max_trg_in_batch, new_data["trg_len"])
    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch

    return max(src_elements, trg_elements)


global src_langs_in_batch, trg_langs_in_batch
def multiparallel_ntokens_limit_langs(new: tuple, count: int, sofar: int) -> int:
    global max_src_in_batch, max_trg_in_batch
    global src_langs_in_batch, trg_langs_in_batch

    new_data = new[1]
    if count == 1:
        src_langs_in_batch = [new_data["src1_lang"], new_data["src2_lang"]]
        trg_langs_in_batch = [new_data["trg_lang"]]
        max_src_in_batch = 0
        max_trg_in_batch = 0

    new_src_langs = [new_data["src1_lang"], new_data["src2_lang"]]
    new_trg_langs = [new_data["trg_lang"]]

    # Check whether languages in example match other languages in batch. If not, start new batch.
    if not (
        all(l in src_langs_in_batch for l in new_src_langs)
        and
        all(l in trg_langs_in_batch for l in new_trg_langs)
    ):
        return math.inf

    # Since we have two source languages that map to the same target, our desired `batch_size` is ~`batch_size` / 2.
    # To account for this, we double the source and target elements.
    max_src_in_batch = max(max_src_in_batch, new_data["src1_len"] + new_data["src2_len"])
    max_trg_in_batch = max(max_trg_in_batch, new_data["trg_len"] * 2)
    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch

    return max(src_elements, trg_elements)

def multiparallel_ntokens(new: tuple, count: int, sofar: int) -> int:
    global max_src_in_batch, max_trg_in_batch

    new_data = new[1]
    if count == 1:
        max_src_in_batch = 0
        max_trg_in_batch = 0

    # Since we have two source languages that map to the same target, our desired `batch_size` is ~`batch_size` / 2.
    # To account for this, we double the source and target elements.
    max_src_in_batch = max(max_src_in_batch, new_data["src1_len"] + new_data["src2_len"])
    max_trg_in_batch = max(max_trg_in_batch, new_data["trg_len"] * 2)
    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch

    return max(src_elements, trg_elements)


global src_langs_in_batch, trg_langs_in_batch
def multiparallel_nsents(new: tuple, count: int, sofar: int) -> int:
    global src_langs_in_batch, trg_langs_in_batch

    new_data = new[1]
    if count == 1:
        src_langs_in_batch = [new_data["src1_lang"], new_data["src2_lang"]]
        trg_langs_in_batch = [new_data["trg_lang"]]

    new_src_langs = [new_data["src1_lang"], new_data["src2_lang"]]
    new_trg_langs = [new_data["trg_lang"]]

    # Check whether languages in example match other languages in batch.
    # If not, start new batch.
    if not (
        all(l in src_langs_in_batch for l in new_src_langs)
        and
        all(l in trg_langs_in_batch for l in new_trg_langs)
    ):
        return math.inf

    return count
