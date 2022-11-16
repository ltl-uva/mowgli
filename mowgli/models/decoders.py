import torch.nn as nn
from torch import Tensor
from mowgli.modules import SinusoidalPositionalEmbedding, TransformerDecoderLayer
from mowgli.helpers import freeze_params, subsequent_mask


class Decoder(nn.Module):
    """Base decoder class."""

    @property
    def output_size(self):
        """Returns the output size (size of the target vocabulary)."""
        return self._output_size

    def __len__(self):
        """Returns the number of layers."""
        return len(self.layers)


class TransformerDecoder(Decoder):
    """Implements Transformer decoder."""

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        vocab_size: int = 1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Initializes a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        """
        super().__init__()

        self._hidden_size = hidden_size
        self._output_size = vocab_size

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size, ff_size=ff_size, num_heads=num_heads, dropout=dropout)
                    for _ in range(num_layers)
            ]
        )

        self.emb_pos = SinusoidalPositionalEmbedding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        embedded_trg: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
    ):
        """
        Transformer decoder forward pass.

        :param embedded_trg: embedded target inputs (batch_size x trg_len x embed_size)
        :param encoder_output: encoded source representations (batch_size * src_len * embed_size)
        :param src_mask: indicates source padding areas, i.e. zeros where padding (batch_size x 1 x src_len)
        :param trg_mask: indicates target padding areas, i.e. zeros where padding (batch_size x 1 x src_len)
        :return:
        """
        x = self.emb_pos(embedded_trg)
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(embedded_trg.size(1)).type_as(trg_mask)

        for i, layer in enumerate(self.layers):
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        decoder_output = self.output_layer(x)

        return decoder_output


    def __repr__(self):
        return f"{self.__class__.__name__}(num_layers={len(self.layers)}, num_heads={self.layers[0].trg_trg_att.num_heads})"
