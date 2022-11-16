import os
from typing import List, Optional
import logging
import numpy as np
import sacrebleu

import torch

from mowgli.helpers import (
    bpe_postprocess,
    make_logger,
    get_latest_checkpoint,
    list2file,
    load_checkpoint,
    load_model_weights,
)
from mowgli.models import build_model, MowgliModel, ModularModel, UniversalModel
from mowgli.decoding import run_batch
from mowgli.data import Batch, build_datasets, build_iterator
from mowgli.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN

logger = logging.getLogger(__name__)


def validate_on_data(
    model: MowgliModel,
    data,
    batch_size: int,
    use_cuda: bool,
    max_output_length: int,
    level: str,
    eval_metric: Optional[str],
    n_gpu: int,
    compute_loss: bool = False,
    save_cross_attn: bool = True,
    beam_size: int = 1,
    beam_alpha: int = -1,
    batch_type: str = "sentence",
    postprocess: bool = True,
    bpe_type: str = "subword-nmt",
    sacrebleu_opt: dict = None,
    num_workers: int = 8,
    rank: int = 0,
) -> dict:
    """
    Generate translations for the given data.
    If `compute_loss` is True and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param n_gpu: number of GPUs
    :param compute_loss: whether to computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param postprocess: if True, remove BPE segmentation from translations
    :param bpe_type: bpe type, one of {"subword-nmt", "sentencepiece"}
    :param sacrebleu_opt: sacrebleu options

    :return:
        - results: dictionary with results
    """
    assert batch_size >= n_gpu, "`batch_size` must be bigger than `n_gpu`."
    model = model if n_gpu < 2 else model.module
    assert isinstance(model, ModularModel) or isinstance(model, UniversalModel)
    decoder = model.decoder if isinstance(model, UniversalModel) else model.decoder[data.trg_lang]
    saves_src_trg_context = model.saves_src_trg_context()
    trg_key = "shared" if model.trg_key == ["shared"] else data.trg_lang
    results = {}
    device = torch.device("cuda" if use_cuda else "cpu", rank)
    results["sources_raw"] = data.src
    cross_attns = {} if save_cross_attn else None
    if sacrebleu_opt is None:
        sacrebleu_opt = {"remove_whitespace": True, "tokenize": "13a"}

    data_iter = build_iterator(
        dataset     = data,
        batch_size  = batch_size,
        batch_type  = batch_type,
        shuffle     = False,
        pad_idx     = model.pad_idx,
        train       = False,
        num_workers = num_workers,
    )

    model.eval()
    model.set_src_trg_context_saving(True)
    with torch.no_grad():
        all_outputs = []
        results["val_attn_scores"] = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        for batch in iter(data_iter):
            batch.to_device(device)

            # Run as during training with teacher forcing
            if compute_loss:
                batch_loss = model(batch)["crossent"]
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

                # Save cross attentions for similarity computation
                if save_cross_attn:
                    for (s_idx, s), dataset_idx in zip(enumerate(batch.trg), batch.idxs):
                        # find first pad token
                        try:
                            stop = (s == model.pad_idx).nonzero(as_tuple=False)[0].item()
                        # if there is no pad token, use full sentence
                        except IndexError:
                            stop = len(s)

                        cross_attns[dataset_idx] = [
                            decoder.layers[i].src_trg_att.context[s_idx][:stop].cpu()
                            for i in range(len(decoder.layers))
                        ]

            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_length()

            # run as during inference to produce translations
            output, attention_scores = run_batch(model, batch, max_output_length, beam_size, beam_alpha)

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            results["val_attn_scores"].extend(attention_scores[sort_reverse_index] if attention_scores is not None else [])

        model.set_src_trg_context_saving(saves_src_trg_context) # return to original state
        assert len(all_outputs) == len(data)
        if save_cross_attn:
            results["cross_attn"] = cross_attns
        if compute_loss and total_ntokens > 0:
            results["loss"] = total_loss
            results["loss_norm"] = total_loss / total_ntokens
            results["ppl"] = torch.exp(total_loss / total_ntokens)
        else:
            results["loss"] = -1
            results["ppl"] = -1

        # decode back to symbols
        results["decoded"] = model.vocab[trg_key].arrays_to_sentences(arrays=all_outputs, cut_at_eos=True)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        results["sources"] = [join_char.join(s) for s in data.src]
        results["refs"]    = [join_char.join(t) for t in data.trg]
        results["hyps"]    = [join_char.join(t) for t in results["decoded"]]

        # post-process
        if level == "bpe" and postprocess:
            results["sources"] = [bpe_postprocess(s, bpe_type=bpe_type) for s in results["sources"]]
            results["refs"]    = [bpe_postprocess(v, bpe_type=bpe_type) for v in results["refs"]]
            results["hyps"]    = [bpe_postprocess(v, bpe_type=bpe_type) for v in results["hyps"]]

        # if references are given, evaluate against them
        if results["refs"]:
            assert len(results["hyps"]) == len(results["refs"])

            results["score"] = 0
            if eval_metric == "bleu":
                results["score"] = sacrebleu.corpus_bleu(
                    sys_stream=results["hyps"], ref_streams=[results["refs"]], tokenize=sacrebleu_opt["tokenize"]
                ).score

            elif eval_metric.lower() == "chrf":
                results["score"] = sacrebleu.corpus_chrf(
                    hypotheses=results["hyps"], references=[results["refs"]], remove_whitespace=sacrebleu_opt["remove_whitespace"]
                ).score
        else:
            results["score"] = -1

    return results

def parse_test_args(cfg, mode="test"):
    """
    parse test args
    :param cfg: config object
    :param mode: 'test' or 'translate'
    :return:
    """
    args = {}

    if "test_path" not in cfg["data"].keys():
        raise ValueError("test data must be specified in config.")

    args["batch_size"] = cfg["training"].get("eval_batch_size", cfg["training"].get("batch_size", 1))
    args["batch_type"] = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    args["use_cuda"] = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    args["eval_metric"] = cfg["training"]["eval_metric"]
    args["level"] = cfg["data"]["level"]
    args["max_output_length"] = cfg["training"].get("max_output_length", None)
    args["n_gpu"] = torch.cuda.device_count() if args["use_cuda"] else 0
    args["num_workers"] = cfg["data"].get("num_workers", 8)
    device = torch.device("cuda" if args["use_cuda"] else "cpu")
    # whether to use beam search for decoding, 0: greedy decodi'sng
    if "testing" in cfg.keys():
        args["beam_size"] = cfg["testing"].get("beam_size", 1)
        args["beam_alpha"] = cfg["testing"].get("alpha", -1)
        args["postprocess"] = cfg["testing"].get("postprocess", True)
        args["bpe_type"] = cfg["testing"].get("bpe_type", "subword-nmt")
        args["sacrebleu_opt"] = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in cfg["testing"].keys():
            args["sacrebleu_opt"]["remove_whitespace"] = cfg["testing"]["sacrebleu"].get("remove_whitespace", True)
            args["sacrebleu_opt"]["tokenize"] = cfg["testing"]["sacrebleu"].get("tokenize", "13a")
    else:
        args["beam_size"] = 1
        args["beam_alpha"] = -1
        args["postprocess"] = True
        args["bpe_type"] = "subword-nmt"
        args["sacrebleu_opt"] = {"remove_whitespace": True, "tokenize": "13a"}

    logger.info(
        f"""Process device: {device},
        n_gpu: {args["n_gpu"]},
        batch_size per device: {args["batch_size"] // args["n_gpu"] if args["n_gpu"] > 1 else args["batch_size"]}"""
    )

    args["decoding_descript"] = (
        "Greedy decoding" if args["beam_size"] < 2
        else
        f"Beam search decoding with beam size = {args['beam_size']} and alpha = {args['beam_alpha']}"
    )

    args["tokenizer_info"] = f"{args['sacrebleu_opt']['tokenize']}" if args["eval_metric"] == "bleu" else ""

    return args


def test(cfg: dict, ckpt: str, output_path: str = None, save_attention: bool = False, datasets: dict = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """
    # remove train / valid data location since we do not need it for testing
    if cfg["data"].get("train"):
        del cfg["data"]["train"]

    model_dir = cfg["training"]["model_dir"]

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    if datasets is None:
        _, _, test_data, vocab, src_key, trg_key = build_datasets(cfg["data"], datasets=["test"])
        data_to_predict = {"test": test_data}
    # avoid to load data again
    else:
        data_to_predict = {"test": datasets["test"]}
        vocab = datasets["vocab"]
        src_key = datasets["src_key"]
        trg_key = datasets["trg_key"]

    # parse test args
    test_args = parse_test_args(cfg, mode="test")

    # load model state from disk
    logger.info(f"Loading '{ckpt}' checkpoint...")
    model_checkpoint = load_checkpoint(ckpt, use_cuda=test_args["use_cuda"])

    # build model and load parameters into it
    model = build_model(cfg=cfg["model"], vocab=vocab, src_key=src_key, trg_key=trg_key)

    load_model_weights(model, model_checkpoint["model_state"])

    if test_args["use_cuda"]:
        model.cuda()

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        for i, pair in enumerate([[pair.src_lang, pair.trg_lang] for pair in data_set]):
            pair = "-".join(pair)
            logger.info(f"({pair}) decoding on {data_set_name} set...")

            results = validate_on_data(
                model,
                data                = data_set[i],
                batch_size          = test_args["batch_size"],
                batch_type          = test_args["batch_type"],
                level               = test_args["level"],
                max_output_length   = test_args["max_output_length"],
                eval_metric         = test_args["eval_metric"],
                use_cuda            = test_args["use_cuda"],
                compute_loss        = False,
                beam_size           = test_args["beam_size"],
                beam_alpha          = test_args["beam_alpha"],
                postprocess         = test_args["postprocess"],
                bpe_type            = test_args["bpe_type"],
                sacrebleu_opt       = test_args["sacrebleu_opt"],
                n_gpu               = test_args["n_gpu"],
                num_workers         = test_args["num_workers"],
            )

            if data_set[i].trg is not None:
                logger.info(
                    f"""{data_set_name} {test_args['eval_metric']}{test_args['tokenizer_info']}:
                    {results['score']:.2f} [{test_args['decoding_descript']}]"""
                )

            else:
                logger.info(f"No references given for {data_set_name} -> no evaluation.")

            if output_path is not None:

                def detokenize(fn, l):
                    """Detokenizes sentence `s` in language `l` through moses detokenizer.perl."""
                    detok_fn = fn.split(".")
                    detok_fn.insert(-1, "detok")
                    detok_fn = ".".join(detok_fn)
                    os.system(f"cat {fn} | ~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l {l} > {detok_fn}")


                list2file(f"{output_path}/{data_set_name}.{pair}.hyps", results["hyps"])
                list2file(f"{output_path}/{data_set_name}.{pair}.refs", results["refs"])

                save_detokenized = True
                if save_detokenized:
                    detokenize(f"{output_path}/{data_set_name}.{pair}.hyps", pair.split("-")[1])
                    detokenize(f"{output_path}/{data_set_name}.{pair}.refs", pair.split("-")[1])

                logger.info(f"Translations saved to: {output_path}/{data_set_name}.{pair}.hyps")
