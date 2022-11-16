from torch import Tensor
from mowgli.models import MowgliModel

from mowgli.data import Batch


class UniversalModel(MowgliModel):
    """Implements universal model. In this type of model, all parameters are shared between all source and target languages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "universal"
        self.n_decoder_layers = len(self.decoder)


    def encode_decode(self, batch: Batch) -> Tensor:
        """
        Feeds the source tokens to the shared encoder, and the encouder output and target tokens to the shared decoder.

        :param batch: parallel batch of source and target data (possibly multiple source/target languages per batch)
        :return: output of decoder (n_sentences x trg_len x vocab_size)
        """
        encoder_output = self.encode(batch.src, batch.src_mask)

        decoder_output = self.decode(
            encoder_output  = encoder_output,
            src_mask        = batch.src_mask,
            trg_input       = batch.trg_input,
            trg_mask        = batch.trg_mask,
        )

        return decoder_output

    def encode(self, src: Tensor, src_mask: Tensor, src_lang: str = None) -> Tensor:
        """
        Feeds source data through shared encoder.

        :param src: source token indices (n_sentences x src_len)
        :param src_mask: masks padding tokens (n_sentences x 1 x src_len)
        :return: encoder output (n_sentences x src_len x hidden_size)
        """
        embedded_src = self.src_embed(src)
        encoder_output = self.encoder(embedded_src=embedded_src, mask=src_mask)

        return encoder_output


    def decode(self, encoder_output: Tensor, src_mask: Tensor, trg_input: Tensor, trg_mask: Tensor, trg_lang: str = None) -> Tensor:
        """
        Feeds encoded source representations and target data through shared decoder.

        :param encoder_output: encoder output (n_sentences x src_len x hidden_size)
        :param trg_input: target token indices (n_sentences x trg_len)
        :param src_mask: masks src padding tokens (n_sentences x 1 x src_len)
        :param trg_mask: masks trg padding tokens (n_sentences x 1 x trg_len)
        :return: decoder output (n_sentences x trg_len x trg_vocab_size)
        """
        embedded_trg = self.trg_embed(trg_input)
        decoder_output = self.decoder(
            embedded_trg    = embedded_trg,
            encoder_output  = encoder_output,
            src_mask        = src_mask,
            trg_mask        = trg_mask,
        )

        return decoder_output
