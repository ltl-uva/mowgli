#!/usr/bin/env python3

import argparse
from collections import OrderedDict
import numpy as np


def build_vocab(output_path, train_paths=None, dataset=None, dataset_path=None, languages=None, min_freq=1):
    """
    Builds the vocabulary.
    Compatible with Nematus build_dict function, but does not
    output frequencies and special symbols.
    :param train_paths:
    :param output_path:
    :return:
    """
    assert not (train_paths and dataset), "Do not provide both `train_paths` and `dataset`. Pick one."
    
    if dataset:
        assert dataset in ["opus100"], "`dataset` {} not supported".format(dataset)
        
        train_paths, seen = [], []
        for l1 in languages:
            for l2 in languages:
                pair = "-".join(sorted([l1, l2]))
                if not l1==l2 and pair not in seen and (l1 == "en" or l2 == "en"):
                    seen.append(pair)
                    train_paths.append(dataset_path+pair+"/opus.spm."+pair+"-train."+l1)
                    train_paths.append(dataset_path+pair+"/opus.spm."+pair+"-train."+l2)                

    counter = OrderedDict()

    # iterate over input paths
    for path in train_paths:
        print("Processing {}...".format(path))
        with open(path, encoding="utf-8", mode="r") as f:
            for line in f:
                for token in line.strip('\r\n ').split(' '):
                    if token:
                        if token not in counter:
                            counter[token] = 0
                        counter[token] += 1

    # remove all tokens with count less than minimum frequency
    counter = {key:value for key, value in counter.items() if value >= min_freq}

    words = list(counter.keys())
    freqs = list(counter.values())
    
    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    with open(output_path, mode='w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + "\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Builds a vocabulary from training file(s)."
                    ""
                    "Can be used to build a joint vocabulary for weight tying."
                    "To do so, first apply BPE to both source and target "
                    "training files, and then build a vocabulary using"
                    "this script from their concatenation."
                    ""
                    "If you provide multiple files then this program "
                    "will merge them before building a joint vocabulary."
                    "")

    ap.add_argument(
        "--train_paths",
        type=str,
        help="One or more input (training) file(s)",
        nargs="+",
        default=None
    )
    ap.add_argument(
        "--dataset",
        type=str,
        help="Default dataset. Currently supports opus100",
        default=None
    )
    ap.add_argument(
        "--dataset_path",
        type=str,
        help="Root path of default dataset",
        default=None
    )
    ap.add_argument(
        "--languages",
        type=str,
        help="Languages to be used for default dataset",
        nargs="+",
        default=None
    )
    ap.add_argument(
        "--output_path",
        type=str,
        help="Output path for the built vocabulary",
        default="vocab.txt"
    )
    ap.add_argument(
        "--min_freq",
        type=int,
        help="Minimum token frequency",
        default=1
    )
    args = ap.parse_args()
    assert args.train_paths or args.dataset, "Provide `train_paths` or `dataset`"

    build_vocab(
        args.output_path,
        train_paths=args.train_paths,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        languages=args.languages,
        min_freq=args.min_freq
    )
