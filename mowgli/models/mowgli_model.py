from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mowgli.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from mowgli.data import Batch, MultiparallelBatch
from mowgli.modules import Embeddings
from mowgli.models import Encoder
from mowgli.models import Decoder
from mowgli.data import Vocabulary


class MowgliModel(nn.Module):
    """
    Implements a generic encoder-decoder model. Can handle parallel and multiparallel batches.
    `encode_decode` function should be overriden by child class.
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embeddings,
        trg_embed: Embeddings,
        vocab: Vocabulary,
        src_key: list,
        trg_key: list,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.vocab = vocab
        self.src_key = src_key
        self.trg_key = trg_key

        self.bos_idx = vocab[self.trg_key[0]].s2i[BOS_TOKEN]
        self.eos_idx = vocab[self.trg_key[0]].s2i[EOS_TOKEN]
        self.pad_idx = vocab[self.trg_key[0]].s2i[PAD_TOKEN]

        # Set by Train Manager (if applicable)
        self.loss_function = None
        self.similarity_loss_function = None
        self.device = None


    def forward(self, batch: Union[Batch, MultiparallelBatch]) -> dict:
        """
        Forward pass for parallel or multiparallel batch through model.

        :param batch: batch of data (parallel or multiparallel) to feed model
        :return: loss dictionary
        """
        assert self.loss_function
        assert batch.type in ["parallel", "multiparallel"]
        return self.parallel_forward(batch) if batch.type == "parallel" else self.multiparallel_forward(batch)


    def parallel_forward(self, batch: Batch) -> dict:
        """
        Forward pass for parallel batch.

        :param batch: parallel batch
        :return: loss dictionary, consisting of crossentropy loss
        """
        decoder_out = self.encode_decode(batch)
        log_probs = F.log_softmax(decoder_out, dim=-1)
        loss = self.loss_function(log_probs, batch.trg)
        return {"crossent": loss}


    def multiparallel_forward(self, batch: MultiparallelBatch) -> dict:
        """
        Forward pass for multiparallel batch.

        :param batch: multiparallel batch
        :return: loss dictionary, consisting of crossentropy loss and (possibly) similarity loss
        """
        if self.similarity_loss_function: cross_attns, src_langs = [], []
        crossent_loss = 0

        # Compute cross-entropy loss for src1->trg and src2->trg
        for b in batch.split_to_parallel():
            b.to_device(self.device)
            crossent_loss += self.forward(b)["crossent"]

            # Save cross attentions for similarity loss
            if self.similarity_loss_function:
                decoder = self.decoder if self.type == "universal" else self.decoder[b.trg_lang]
                cross_attn = [decoder.layers[i].src_trg_att.context for i in range(self.n_decoder_layers)]
                cross_attns.append(cross_attn)
                src_langs.append(b.src_lang)


        # If available, compute similarity loss between src1->trg and src2->trg cross attention distributions
        sim_loss = None
        if self.similarity_loss_function:
            sim_loss = self.similarity_loss_function(
                x1      = cross_attns[0],
                x2      = cross_attns[1],
                lang1   = src_langs[0],
                lang2   = src_langs[1],
                mask    = batch.trg.eq(self.pad_idx),
            )

        return {"crossent": crossent_loss, "similarity": sim_loss}


    def encode_decode(**kwargs):
        """Should be overriden by child class."""
        raise NotImplementedError


    def set_loss_function(self, loss_function: Callable):
        """Sets the loss function."""
        assert not self.loss_function
        self.loss_function = loss_function


    def set_similarity_loss_function(self, similarity_loss_function: Callable):
        """Sets the similarity loss function. Only called when optimizing similarity."""
        assert not self.similarity_loss_function
        self.similarity_loss_function = similarity_loss_function


    def set_device(self, device: torch.device):
        """Sets the device (cpu or gpu)."""
        assert not self.device
        self.device = device


    def saves_src_trg_context(self) -> bool:
        """Returns whether model stores source target context attention vectors."""        
        decoder = self.decoder[list(self.decoder.keys())[0]] if self.type == "modular" else self.decoder
        return decoder.layers[0].src_trg_att._save_src_trg_context


    def set_src_trg_context_saving(self, save_src_trg_context: bool):
        """If set to `True`, model stores source target context (cross-attention)."""
        if self.type == "modular":
            for language in self.decoder.keys():
                for layer in self.decoder[language].layers:
                    layer.src_trg_att._save_src_trg_context = save_src_trg_context

        elif self.type == "universal":
            for layer in self.decoder.layers:
                layer.src_trg_att._save_src_trg_context = save_src_trg_context



    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tencoder={self.encoder},\n"
            f"\tdecoder={self.decoder},\n"
            f"\tsrc_embed={self.src_embed},\n"
            f"\ttrg_embed={self.trg_embed})"
        )
