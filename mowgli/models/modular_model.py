import torch
from torch import Tensor

from mowgli.data import Batch
from mowgli.models import MowgliModel


class ModularModel(MowgliModel):
    """
    Implements modular model. In this type of model, all source languages have separate encoders, and all target languages
    have different decoders.
    """

    def __init__(self, **kwargs):
        assert len(kwargs["trg_key"]) == 1, "Only many-to-one is supported for now."
        super().__init__(**kwargs)
        self.type = "modular"
        self.n_decoder_layers = len(self.decoder[self.trg_key[0]])


    def encode_decode(self, batch: Batch) -> Tensor:
        """
        Feeds the source tokens to the corresponding encoder, and the encouder output and target tokens to the corresponding
        decoder. Only single source language and single target language allowed per batch.

        :param batch: parallel batch of source and target data
        :return: output of decoder (n_sentences x trg_len x vocab_size)
        """
        encoder_output = self.encode(src=batch.src, src_lang=batch.src_lang, src_mask=batch.src_mask)

        decoder_output = self.decode(
            encoder_output  = encoder_output,
            trg_input       = batch.trg_input,
            trg_lang        = batch.trg_lang,
            src_mask        = batch.src_mask,
            trg_mask        = batch.trg_mask,
        )

        return decoder_output


    def encode(self, src: Tensor, src_mask: Tensor, src_lang: str) -> Tensor:
        """
        Feeds source data through corresponding encoder.

        :param src: source token indices (n_sentences x src_len)
        :param src_lang: source language, used to map data to correct encoder
        :param src_mask: masks padding tokens (n_sentences x 1 x src_len)
        :return: encoder output (n_sentences x src_len x hidden_size)
        """
        embedded_src = self.src_embed[src_lang](src)
        encoder_output = self.encoder[src_lang](embedded_src=embedded_src, mask=src_mask)

        return encoder_output


    def decode(self, encoder_output: Tensor, trg_input: Tensor, src_mask: Tensor, trg_mask: Tensor, trg_lang: str) -> Tensor:
        """
        Feeds encoded source representations and target data through corresponding decoder.

        :param encoder_output: encoder output (n_sentences x src_len x hidden_size)
        :param trg_input: target token indices (n_sentences x trg_len)
        :param trg_lang: target language, used to map data to correct decoder
        :param src_mask: masks src padding tokens (n_sentences x 1 x src_len)
        :param trg_mask: masks trg padding tokens (n_sentences x 1 x trg_len)
        :return: decoder output (n_sentences x trg_len x trg_vocab_size)
        """
        embedded_trg = self.trg_embed[trg_lang](trg_input)
        decoder_output = self.decoder[trg_lang](
            embedded_trg    = embedded_trg,
            encoder_output  = encoder_output,
            src_mask        = src_mask,
            trg_mask        = trg_mask,
        )

        return decoder_output
