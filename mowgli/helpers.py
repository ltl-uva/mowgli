import copy
from functools import reduce
import glob
import os
import os.path
import operator
import errno
import io
import shutil
import random
import re
import itertools
import logging
import time
from typing import Optional, List
import numpy as np
import pkg_resources
import yaml

import torch
from torch import nn, Tensor


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def make_model_dir(model_dir: str, overwrite=False, rank=0) -> str:
    """
    Creates a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    # only rank 0 creates a model directory
    if rank != 0:
        # hack to ensure that logger for higher ranks is not deleted
        time.sleep(3)
        return model_dir

    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(f"Model directory {model_dir} exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(os.path.join(model_dir, "translations"))

    return model_dir


def make_logger(log_dir: str = None, mode: str = "train", rank=0) -> str:
    """
    Create a logger for logging the training/testing process.

    :param log_dir: path to file where log is stored as well
    :param mode: log file name. 'train', 'test' or 'translate'
    :return: mowgli version number
    """
    logger = logging.getLogger("") # root logger
    version = pkg_resources.require("mowgli")[0].version

    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )

        if log_dir is not None:
            if os.path.exists(log_dir):
                log_file = '{}/{}_rank_{}.log'.format(log_dir, mode, rank)

                # write to file
                fh = logging.FileHandler(log_file)
                fh.setLevel(level=logging.DEBUG)
                logger.addHandler(fh)
                fh.setFormatter(formatter)

        # write to stdout (only rank 0)
        if rank == 0:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logger.addHandler(sh)

        logger.info("Hello! This is Mowgli NMT (version %s).", version)

    return version


def flatten(l):
    try:              return [item for sublist in l for item in sublist]
    except TypeError: return l # if l is already flattened return l


def log_cfg(cfg: dict, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param prefix: prefix for logging
    """
    logger = logging.getLogger(__name__)
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def log_data_info(
    train_data,
    valid_data,
    test_data,
    vocab,
    src_key,
    trg_key,
) -> None:
    """
    Log statistics of data and vocabulary.
    """
    logger = logging.getLogger(__name__)

    logger.info("Data set sizes: \n\ttrain {},\n\tvalid {},\n\ttest {}".format(
        sum([len(data) for data in train_data]),
        sum([len(data) for data in valid_data]),
        sum([len(data) for data in test_data] ) if test_data is not None else 0)
    )
    for data in train_data:
        if data.type == "parallel":
            logger.info("First training example (parallel):\n\t{SRC} \n\t{TRG} ".format(
                SRC=" ".join([vocab[src_key[0]].i2s[w] for w in data[0]["src"]]),
                TRG=" ".join([vocab[trg_key[0]].i2s[w] for w in data[0]["trg"]]))
            )
        elif data.type == "multiparallel":
            src1_lang = data[0]["src1_lang"]
            src2_lang = data[0]["src2_lang"]
            trg_lang  = data[0]["trg_lang"]
            logger.info("1st training example (multiparallel):\n\t{} \n\t{} \n\t{} ".format(
                " ".join([vocab[src_key[0]].i2s[w] for w in data[0]["src1"]]),
                " ".join([vocab[src_key[0]].i2s[w] for w in data[0]["src2"]]),
                " ".join([vocab[trg_key[0]].i2s[w] for w in data[0]["trg"]]))
            )

    for k in set(src_key + trg_key):
        logger.info("First 10 words ({}): {}".format(k, [(i, t) for i, t in enumerate(vocab[k].i2s[:10])]))
        logger.info("Number of words ({}): {}".format(k, len(vocab[k])))


def load_config(path, override_args=None) -> dict:
    """
    Loads and parses a YAML configuration file. Overrides YAML values with command prompt input where applicable.

    :param path: path to YAML configuration file
    :param override_args: to be changed arguments
    :return: configuration dictionary
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    def getFromDict(dataDict, mapList):
        """Returns value from dict. Allows indexing through `.` instead of []"""
        return reduce(operator.getitem, mapList, dataDict)

    def setInDict(dataDict, mapList, value):
        """Sets value in dict. Allows indexing through `.` instead of []"""
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

    str2list = lambda x: list(map(str.strip, x.strip('][').replace('"', '').split(',')))

    is_true         = lambda x: x == "True"
    is_false        = lambda x: x == "False"
    is_none         = lambda x: x == "None"
    is_float        = lambda x: bool(re.match(r'^-?\d+(?:\.\d+)$', x))
    is_int          = lambda x: x.isdigit()
    is_list         = lambda x: x[0] == "[" and x[-1] == "]"
    is_int_list     = lambda x: is_list(x) and all([is_int(v) for v in x[1:-1].split(",")])
    is_float_list   = lambda x: is_list(x) and all([is_float(v) for v in x[1:-1].split(",")])
    is_str_list     = lambda x: is_list(x)

    if override_args is not None:
        for override in override_args:
            override_key, override_val = override.split(":")

            # Convert string to correct type
            if   is_true(override_val):         override_val = True
            elif is_false(override_val):        override_val = False
            elif is_none(override_val):         override_val = None
            elif is_float(override_val):        override_val = float(override_val)
            elif is_int(override_val):          override_val = int(override_val)
            elif is_float_list(override_val):   override_val = [float(v) for v in str2list(override_val)]
            elif is_int_list(override_val):     override_val = [int(v) for v in str2list(override_val)]
            elif is_str_list(override_val):     override_val = str2list(override_val)

            # Update value in config dict
            setInDict(cfg, override_key.split("."), override_val)

    return cfg


def bpe_postprocess(string, bpe_type="subword-nmt") -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :param bpe_type: one of {"sentencepiece", "subword-nmt"}
    :return: post-processed string
    """
    if bpe_type == "sentencepiece": ret = string.replace(" ", "").replace("▁", " ").strip()
    elif bpe_type == "subword-nmt": ret = string.replace("@@ ", "").strip()
    else:                           ret = string.strip()

    return ret


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError("No checkpoint found in directory {}."
                                .format(ckpt_dir))
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True, logger=None) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    if logger: logger.info("Loading checkpoint from {}...".format(string))
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


def load_model_weights(model: nn.Module, model_state: dict):
    """
    Load model weights from model state.
    Handles mapping of DDP model to single device.

    :param model: initialized model
    :param model_state: dict with to be restored module names and weights
    """
    # Rename keys (remove `module`) when DDP model is loaded on single device
    if all([m.split(".")[0] == "module" for m in model_state.keys()]):
        model_state = {".".join(k.split(".")[1:]): v for k, v in model_state.items()}

    model.load_state_dict(model_state)


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def file2list(file, reduce_size=None):
    shorten = False if reduce_size is None else True
    lines = []
    with io.open(file, mode='r', encoding='utf-8') as text:
        for i, line in enumerate(text):
            lines.append(line.rstrip())
            if shorten and i+1 == reduce_size:
                break
    return lines

def list2file(fn, l, encoding="utf-8"):
    with open(fn, mode="w", encoding=encoding) as out_file:
        for s in l:
            out_file.write(s + "\n")


def merge_iterators(iterators: list, batch_multiplier=1, datasets=None):
    """Merges iterators so that they take turns yielding a batch. The shortest iterator is upsampled to ensure equal ratio."""
    assert len(iterators) == 1 or len(iterators) == 2

    # Get number of iterations and find largest iterator(s)
    try:
        iterator_lengths = [len(it) for it in iterators]
    except NotImplementedError:
        # if len() is not implemented, get total number of target tokens as proxy for number of batches
        if datasets is not None:
            iterator_lengths = [d.total_tokens for d in datasets]
        # # or generate all batches (this should be avoided, it is very inefficient)
        else:
            iterator_lengths = [len([b for b in it]) for it in iterators]

    # repeat shortest
    iterators = [
        itertools.cycle(it) if idx != iterator_lengths.index(max(iterator_lengths)) else it
        for idx, it in enumerate(iterators)
    ]

    mem = []
    # Yield data from iterators. Do not cycle (upsample) since data is shuffled at every epoch, so eventually all data is seen.
    for batches in zip(*iterators):
        # If batch not completed yet, store in memory
        if len(mem) < batch_multiplier:
            mem.append(batches)

        # If full batch, yield batch
        else:
            for i in range(len(mem[0])):
                for b in mem:
                    yield b[i]
            mem = [batches]

    # At the end of epoch, empty memory
    for i in range(len(mem[0])):
        for b in mem:
            yield b[i]
