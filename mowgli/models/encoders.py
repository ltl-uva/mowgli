import torch.nn as nn
from torch import Tensor

from mowgli.helpers import freeze_params
from mowgli.modules import TransformerEncoderLayer, SinusoidalPositionalEmbedding


class Encoder(nn.Module):
    """Base encoder class."""

    @property
    def output_size(self):
        """Returns the output size."""
        return self._output_size

    def __len__(self):
        """Returns the number of layers."""
        return len(self.layers)


class TransformerEncoder(Encoder):
    """Implements Transformer Encoder."""

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Initializes a Transformer encoder.

        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size. (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        self._output_size = hidden_size

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(size=hidden_size, ff_size=ff_size, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.emb_pos = SinusoidalPositionalEmbedding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        if freeze:
            freeze_params(self)


    def forward(self, embedded_src: Tensor, mask: Tensor) -> Tensor:
        """
        Feeds embedded source tokens through Transformer encoder layers. Returns encoder output.

        :param embedded_src: embedded source inputs (batch_size x src_len x embed_size)
        :param mask: indicates padding areas, i.e. zeros where padding (batch_size x src_len x embed_size)
        :return: last hidden state (batch_size x directions*hidden)
        """
        x = self.emb_pos(embedded_src)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)


    def __repr__(self):
        return f"{self.__class__.__name__}(num_layers={len(self.layers)}, num_heads={self.layers[0].src_src_att.num_heads})"
