# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from mowgli.data import Batch, MultiparallelBatch
from mowgli.models import TransformerDecoder
from mowgli.models import MowgliModel, ModularModel, UniversalModel
from mowgli.helpers import tile

__all__ = ["greedy", "beam_search", "run_batch", "run_multiparallel_batch"]

def greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: MowgliModel,
    encoder_output: Tensor,
    trg_lang: str,
) -> (np.array, None):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    bos_index = model.bos_idx
    eos_index = model.eos_idx
    batch_size = src_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])

    finished = src_mask.new_zeros(batch_size).byte()

    for _ in range(max_output_length):
        # pylint: disable=unused-variable
        with torch.no_grad():
            logits = model.decode(
                encoder_output  = encoder_output,
                src_mask        = src_mask,
                trg_mask        = trg_mask,
                trg_input       = ys, # model.trg_embed(ys) # embed the previous tokens
                trg_lang        = trg_lang,
            )

            logits = logits[:, -1]
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    ys = ys[:, 1:]  # remove BOS-symbol
    return ys.detach().cpu().numpy(), None

def beam_search(
    model: MowgliModel,
    size: int,
    encoder_output: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    trg_lang: str,
    n_best: int = 1,
) -> (np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param model:
    :param size: size of the beam
    :param encoder_output:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert size > 0, 'Beam size must be > 0.'
    assert n_best <= size, 'Can only return {} best hypotheses.'.format(size)

    # init
    bos_index = model.bos_idx
    eos_index = model.eos_idx
    pad_idx = model.pad_idx
    decoder = model.decoder if isinstance(model, UniversalModel) else model.decoder[trg_lang]
    trg_vocab_size = decoder.output_size
    device = encoder_output.device
    batch_size = src_mask.size(0)

    # batch*k x src_len x enc_hidden_size
    encoder_output = tile(encoder_output.contiguous(), size, dim=0)
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    # create target mask
    trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # numbering elements in the extended batch, i.e. beam size copies of each batch element
    beam_offset = torch.arange(0, batch_size*size, step=size, dtype=torch.long, device=device)

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full([batch_size * size, 1], bos_index, dtype=torch.long, device=device)

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.zeros(batch_size, size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):
        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        decoder_input = alive_seq  # complete prediction so far

        # expand current hypotheses
        # decode one single step
        # logits: logits for final softmax
        # pylint: disable=unused-variable
        with torch.no_grad():
            logits = model.decode(
                encoder_output  = encoder_output,
                src_mask        = src_mask,
                trg_mask        = trg_mask,
                trg_input       = decoder_input,
                trg_lang        = trg_lang,
            )

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        logits = logits[:, -1]  # keep only the last time step

        # batch*k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * trg_vocab_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        # topk_beam_index = torch.div(topk_ids, trg_vocab_size, rounding_mode="floor")
        topk_beam_index = topk_ids.floor_divide(trg_vocab_size)
        topk_ids = topk_ids.fmod(trg_vocab_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(True)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    # Check if the prediction has more than one EOS.
                    # If it has more than one EOS, it means that the
                    # prediction should have already been added to
                    # the hypotheses, so you don't have to add them again.
                    if (predictions[i, j, 1:] == eos_index).nonzero(
                            as_tuple=False).numel() < 2:
                        # ignore start_token
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(False).nonzero(
                as_tuple=False).view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)


    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_idx)

    return final_outputs, None


def run_batch(model: MowgliModel, batch: Batch, max_output_length: int,
              beam_size: int, beam_alpha: float) -> (np.array, np.array):
    """
    Get outputs and attentions scores for a given batch

    :param model: Model class
    :param batch: batch to generate hypotheses for
    :param max_output_length: maximum length of hypotheses
    :param beam_size: size of the beam for beam search, if 0 use greedy
    :param beam_alpha: alpha value for beam search
    :return: stacked_output: hypotheses for batch,
        stacked_attention_scores: attention scores for batch
    """
    with torch.no_grad():
        encoder_output = model.encode(src=batch.src, src_mask=batch.src_mask, src_lang=batch.src_lang)

    # if maximum output length is not globally specified, adapt to source length
    if max_output_length is None:
        max_output_length = int(max(batch.src_length.cpu().numpy()) * 1.5)

    # greedy decoding
    if beam_size < 2:
        stacked_output, stacked_attention_scores = greedy(
            src_mask            = batch.src_mask,
            max_output_length   = max_output_length,
            model               = model,
            encoder_output      = encoder_output,
            trg_lang            = trg_lang,
        )
        # batch, time, max_src_length
    else:  # beam search
        stacked_output, stacked_attention_scores = beam_search(
            model               = model,
            size                = beam_size,
            encoder_output      = encoder_output,
            src_mask            = batch.src_mask,
            max_output_length   = max_output_length,
            trg_lang            = batch.trg_lang,
            alpha               = beam_alpha,
        )

    return stacked_output, stacked_attention_scores


def run_multiparallel_batch(
    model: MowgliModel,
    batch: MultiparallelBatch,
    max_output_length: int,
    beam_size: int,
    beam_alpha: float
):
    """
    Get outputs and attention scores for batch with multiple sources and single target
    """
    parallel_batch1 = Batch(batch.src1, batch.trg, batch.src1_length, batch.trg_length, model.pad_idx)
    parallel_batch2 = Batch(batch.src2, batch.trg, batch.src2_length, batch.trg_length, model.pad_idx)

    output_src1, attn_src1 = run_batch(model, parallel_batch1, max_output_length, beam_size, beam_alpha)
    output_src2, attn_src2 = run_batch(model, parallel_batch2, max_output_length, beam_size, beam_alpha)

    return [output_src1, output_src2], [attn_src1, attn_src2]
